/* Copyright 2016 Kristoffer Bernhem.
 * This file is part of SMLocalizer.
 *
 *  SMLocalizer is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  SMLocalizer is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with SMLocalizer.  If not, see <http://www.gnu.org/licenses/>.
 */
/**
 *
 * @author kristoffer.bernhem@gmail.com
 */
import static jcuda.driver.JCudaDriver.cuCtxCreate;
import static jcuda.driver.JCudaDriver.cuCtxSynchronize;
import static jcuda.driver.JCudaDriver.cuDeviceGet;
import static jcuda.driver.JCudaDriver.cuInit;
import static jcuda.driver.JCudaDriver.cuLaunchKernel;
import static jcuda.driver.JCudaDriver.cuMemFree;
import static jcuda.driver.JCudaDriver.cuMemcpyDtoH;
import static jcuda.driver.JCudaDriver.cuModuleGetFunction;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import ij.ImagePlus;
import ij.process.ImageProcessor;
import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.CUcontext;
import jcuda.driver.CUdevice;
import jcuda.driver.CUdeviceptr;
import jcuda.driver.CUfunction;
import jcuda.driver.CUmodule;
import jcuda.driver.JCudaDriver;
import static jcuda.driver.JCudaDriver.cuModuleLoadDataEx;

/* This class contains all relevant algorithms for background corrections. Handles 2D and 3D stacks with single slice per frame.
 * 
 */

class BackgroundCorrection {

	/* Main function for background filtering using the time median based method described in:
	 * The fidelity of stochastic single-molecule super-resolution reconstructions critically depends upon robust background estimation
	 *	E. Hoogendoorn, K. C. Crosby, D. Leyton-Puig, R. M.P. Breedijk, K. Jalink, T. W.J. Gadella & M. Postma
	 *	Scientific Reports 4, Article number: 3854 (2014)	
	 */
	public static void medianFiltering(final int[] W, ImagePlus image, int selectedModel){		
		int nChannels 	= image.getNChannels();
		int nFrames 	= image.getNFrames();
		if (nFrames == 1)
			nFrames = image.getNSlices();  				// some formats store frames as slices, some as frames.
		int columns = image.getWidth();
		int rows	= image.getHeight();		
		if(selectedModel == 0) // parallel.
		{
			for (int Ch = 1; Ch <= nChannels; Ch++){ // Loop over all channels.
				float[] MeanFrame = new float[nFrames]; 		// Will include frame mean value.
				float[][] timeVector = new float[rows*columns][nFrames];
				ImageProcessor IP = image.getProcessor();		// get image processor for the stack.´


				for (int Frame = 1; Frame < nFrames+1; Frame++){			
					if (image.getNFrames() == 1)
					{
						image.setPosition(							
								Ch,			// channel.
								Frame,			// slice.
								1);		// frame.
					}
					else
					{														
						image.setPosition(
								Ch,			// channel.
								1,			// slice.
								Frame);		// frame.
					}
					IP = image.getProcessor(); 			// Update processor to next slice.


					for (int i = 0; i < rows*columns; i++)
					{
						MeanFrame[Frame-1] += IP.get(i);
					}
					MeanFrame[Frame-1] /= rows*columns;

					if (MeanFrame[Frame-1] < 1)
						MeanFrame[Frame-1] = 1;
					for (int i = 0; i < rows*columns; i++)
					{
						timeVector[i][Frame-1] = (float) (IP.get(i )/MeanFrame[Frame-1]); // load data. 						

					}
				} // Data loading.

				// calculate steplength for median filtering:
				int stepLength = nFrames/300;
				if (stepLength > 10)
					stepLength = 10;
				if(nFrames < 500)
					stepLength = 1;
				
				List<Callable<float[]>> tasks = new ArrayList<Callable<float[]>>();	// Preallocate.
				if (stepLength == 1){
					for (int i = 0; i < rows*columns; i++)
					{
						final float[] timeVectorF = timeVector[i];

						final int chFinal = Ch - 1;
						Callable<float[]> c = new Callable<float[]>() {				// Computation to be done.
							@Override
							public float[] call() throws Exception {
								return runningMedian(timeVectorF, W[chFinal]);						// Actual call for each parallel process.
							}
						};
						tasks.add(c);
					}
				}else // if we're speeding up computation.
				{
					for (int i = 0; i < rows*columns; i++)
					{
						final float[] timeVectorF = timeVector[i];

						final int chFinal = Ch - 1;
						final int steps = stepLength; 
						Callable<float[]> c = new Callable<float[]>() {				// Computation to be done.
							@Override
							public float[] call() throws Exception {
								return runningMedian(timeVectorF, W[chFinal],steps);						// Actual call for each parallel process.
							}
						};
						tasks.add(c);
					}
				}
				int processors 			= Runtime.getRuntime().availableProcessors();	// Number of processor cores on this system.
				ExecutorService exec 	= Executors.newFixedThreadPool(processors);		// Set up parallel computing using all cores.
				try {
					List<Future<float[]>> parallelCompute = exec.invokeAll(tasks);				// Execute computation.    
					for (int i = 0; i < parallelCompute.size(); i++){							// Loop over and transfer results.
						try {
							float[] data = parallelCompute.get(i).get();

							for (int k = 0; k < data.length; k++){	
								timeVector[i][k] = (int)(data[k]*MeanFrame[k]);								
							}			
						} catch (ExecutionException e) {
							e.printStackTrace();
						}
					}

				} catch (InterruptedException e) {

					e.printStackTrace();
				}
				finally {
					exec.shutdown();
				}
				int value = 0;
				for (int Frame = 1; Frame <= nFrames; Frame++) // store data.
				{						
					if (image.getNFrames() == 1)
					{
						image.setPosition(							
								Ch,			// channel.
								Frame,			// slice.
								1);		// frame.
					}
					else
					{														
						image.setPosition(
								Ch,			// channel.
								1,			// slice.
								Frame);		// frame.
					}
					IP 	= image.getProcessor(); 			// Update processor to next slice.
					for (int i = 0; i < rows*columns; i++)
					{
						value = (int)timeVector[i][Frame-1];											
						IP.set(i , value);	
					}
				}
				image.updateAndDraw();
			} // Channel loop.

		}else // end parallel.
			if(selectedModel == 2) // GPU.
			{					
				JCudaDriver.setExceptionsEnabled(true);
				// Initialize the driver and create a context for the first device.
				cuInit(0);
				CUdevice device = new CUdevice();
				cuDeviceGet(device, 0);
				CUcontext context = new CUcontext();
				cuCtxCreate(context, 0, device);
				// Load the PTX that contains the kernel.
				CUmodule module = new CUmodule();
				
				String ptxFileName = "medianFilter.ptx";
				byte ptxFile[] = CUDA.loadData(ptxFileName);
//				cuModuleLoad(module, "medianFilter.ptx"); // old  version, loading directly from the ptx file.
				cuModuleLoadDataEx(module, Pointer.to(ptxFile), 
			            0, new int[0], Pointer.to(new int[0]));
				// Obtain a handle to the kernel function.
				CUfunction function = new CUfunction();
				cuModuleGetFunction(function, module, "medianKernel");
				long GB = 1024*1024*1024;
				int frameSize = (3*columns*rows)*Sizeof.FLOAT;

				for(int Ch = 1; Ch <= nChannels; Ch++)
				{
					int staticMemory = (2*W[Ch-1]+1)*rows*columns*Sizeof.FLOAT;
					long framesPerBatch = (3*GB-staticMemory)/frameSize; // 3 GB memory allocation gives this numbers of frames. 					
					if (framesPerBatch > 10000) // longer vectors means longer processing times and will result in timeout error.
						framesPerBatch = 10000;
					int loadedFrames = 0;
					int startFrame = 1;					
					int endFrame = (int)framesPerBatch;					
					if (endFrame > nFrames)
						endFrame = nFrames;

					CUdeviceptr device_window 		= CUDA.allocateOnDevice((float)((2 * W[Ch-1] + 1) * rows * columns)); // swap vector.
					while (loadedFrames < nFrames-1)
					{		
						float[] timeVector = new float[(endFrame-startFrame+1) * rows * columns];
						float[] MeanFrame = new float[endFrame-startFrame+1]; 				// Will include frame mean value.
						ImageProcessor IP = image.getProcessor();
						int frameCounter = 0;
						for (int Frame = startFrame; Frame <= endFrame; Frame++)
						{			
							if (image.getNFrames() == 1)
							{
								image.setPosition(							
										Ch,			// channel.
										Frame,			// slice.
										1);		// frame.
							}
							else
							{														
								image.setPosition(
										Ch,			// channel.
										1,			// slice.
										Frame);		// frame.
							}
							IP = image.getProcessor();

							for (int i = 0; i < rows*columns; i ++)
							{
								timeVector[frameCounter + (endFrame-startFrame+1)*i] = IP.get(i);			
								MeanFrame[frameCounter] += IP.get(i);
							}
							MeanFrame[frameCounter] /= (columns*rows);
							loadedFrames++;
							frameCounter++;
						} // frame loop for mean calculations.
						
						int stepLength = nFrames/300;
						if (stepLength > 10)
							stepLength = 10;
						if(nFrames < 500)
							stepLength = 1;
						CUdeviceptr device_Data 		= CUDA.copyToDevice(timeVector);
						CUdeviceptr device_meanVector 	= CUDA.copyToDevice(MeanFrame);
						CUdeviceptr deviceOutput 		= CUDA.allocateOnDevice((int)timeVector.length);

						int filterWindowLength 		= (2 * W[Ch-1] + 1) * rows * columns;
						int testDataLength 			= timeVector.length;
						int meanVectorLength 		= MeanFrame.length;
						Pointer kernelParameters 	= Pointer.to(   
								Pointer.to(new int[]{W[Ch]}),
								Pointer.to(device_window),
								Pointer.to(new int[]{filterWindowLength}),
								Pointer.to(new int[]{(meanVectorLength)}),
								Pointer.to(device_Data),
								Pointer.to(new int[]{testDataLength}),
								Pointer.to(device_meanVector),
								Pointer.to(new int[]{meanVectorLength}),								
								Pointer.to(new int[]{stepLength}),
								Pointer.to(deviceOutput),
								Pointer.to(new int[]{testDataLength})
								);
						

						int blockSizeX 	= 1;
						int blockSizeY 	= 1;				   
						int gridSizeX 	= columns;
						int gridSizeY 	= rows;
						cuLaunchKernel(function,
								gridSizeX,  gridSizeY, 1, 	// Grid dimension
								blockSizeX, blockSizeY, 1,  // Block dimension
								0, null,               		// Shared memory size and stream
								kernelParameters, null 		// Kernel- and extra parameters
								);
						cuCtxSynchronize();

						// Pull data from device.
						int hostOutput[] = new int[timeVector.length];
						cuMemcpyDtoH(Pointer.to(hostOutput), deviceOutput,
								timeVector.length * Sizeof.INT);


						// return data.		
						int idx = 0;
						for (int Frame = startFrame; Frame <= endFrame; Frame++)
						{			
							if (image.getNFrames() == 1)
							{
								image.setPosition(							
										Ch,			// channel.
										Frame,		// slice.
										1);		// frame.
							}
							else
							{														
								image.setPosition(
										Ch,			// channel.
										1,			// slice.
										Frame);		// frame.
							}
							IP = image.getProcessor();		
							for (int i = 0; i < rows*columns; i ++)
							{			
							//	IP.set(i, hostOutput[idx]);

								if (hostOutput[idx] > 0)
									IP.set(i, (int)hostOutput[idx]);
								else
									IP.set(i, (int)0);
								idx++;
							}

							image.setProcessor(IP);
						} // frame loop for data return.

						startFrame = endFrame-W[Ch-1]; // include W more frames to ensure that border errors from median calculations dont occur ore often then needed.
						endFrame += framesPerBatch;					
						if (endFrame > nFrames)
							endFrame = nFrames;
						
						// Free up memory allocation on device, housekeeping.

						cuMemFree(device_Data);    
						cuMemFree(deviceOutput);
						cuMemFree(device_meanVector);
					} // while loadedChannels < nFrames
					// Free up memory allocation on device, housekeeping.
					cuMemFree(device_window);
				} // Channel loop.				
				image.updateAndDraw();
			} // end GPU.
	} // medianfiltering with image output.


	
	/*
	 * Test function
	 * 
	 */
	public static void main(String[] args)
	{
		double[] temp = {604, 810, 789, 753, 653, 717, 619, 736, 571, 688, 720, 804, 906, 663, 674, 883, 740, 754, 658, 599
                , 869, 689, 616, 746, 625, 676, 765, 639, 689, 713, 674, 659, 697, 822, 693, 638, 759, 750, 706, 720, 
                728, 672, 530, 543, 675, 797, 666, 916, 615, 725, 537, 588, 625, 715, 757, 786, 608, 650, 687, 774};
		float[] data = new float[temp.length];
		for(int i = 0; i < data.length; i++)
			data[i] = (float) temp[i];		
	/*	long time = System.nanoTime();
		for(int i = 0; i < 10; i ++)
			runningMedian(data,50);
		long time2 = System.nanoTime();		
		for(int i = 0; i < 10; i ++)
			runningMedian(data,50,4);
		long time3 = System.nanoTime();
		System.out.println("old:" + (time2-time)*1E-7 + " vs " + (time3 - time2)*1E-7 + " for total: "+ (float)((time2-time)/(time3-time2)));


*/
	//	float[] old = runningMedian(data,50);
		float[] New = runningMedian(data,10,4);
	
		
		for (int i = 0; i < 150;i++)
		{
			if (i%10 == 0)
				System.out.println(" ");
			System.out.print(New[i] + " ");
		}
		
	}


	/*
	 * float precision version skipping skipNr and interpolating. Performance enhancement based on size of skipNr, ranging from 4-10x. Shown by original publication to be accurate for up to stepsize of 10.
	 */
	public static float[] runningMedian(float[] Vector, int W, int skipNr){
		// Preallocate variables.				
		float[] medianVector = new float[Vector.length]; // Output vector.
		float[] V = new float[2*W+1];  // Vector for calculating running median.		

		int low  = 0;
		int high = W;			   
		int idx = 0;

		while (idx <= W*skipNr)
		{
			V[idx/skipNr] = Vector[idx];
			idx += skipNr;
		}
		quickSort(V,low,high);
		// V now sorted and populated with first W+1 elements taken from Vector at every skipNr index.
		high++;
		//		idx+=skipNr;
		medianVector[0] = V[W/2];
		int inpIdx = skipNr;
		while (idx <  (2*W+1)*skipNr && idx < Vector.length - W*skipNr)
		{
			//System.out.println("high: " + high + " idx: " + idx);
			V[high] = Vector[idx]; // next element.
			quickSort(V,low,high);
			medianVector[inpIdx] = (V[high/2] + (high%2)*V[high/2 + 1])/(1+high%2);			
			float step = medianVector[inpIdx] - medianVector[inpIdx-skipNr];
			step /= skipNr;
			for (int i = 1; i < skipNr; i++)
			{
				medianVector[inpIdx-skipNr + i] = medianVector[inpIdx-skipNr] + i*step;

			}
			high++;
			idx+=skipNr;	
			inpIdx+=skipNr;
		}
		while (idx < Vector.length) // main loop.
		{
			replaceEntry(V,Vector[idx-(2*W+1)*skipNr],Vector[idx]);
			medianVector[inpIdx] = V[high/2];
			float step = medianVector[inpIdx] - medianVector[inpIdx-skipNr];
			step /= skipNr;
			for (int i = 1; i < skipNr; i++)
			{
				medianVector[inpIdx-skipNr + i] = medianVector[inpIdx-skipNr] + i*step;

			}
			idx+=skipNr;
			inpIdx+=skipNr;
		}

		while (inpIdx < Vector.length) // final portion. Remove elements and calculate median on remainder. 
		{
			V = removeEntry(V,Vector[idx-(2*W+1)*skipNr]);
			medianVector[inpIdx] = (V[V.length/2] + (V.length%2)*V[V.length/2 + 1])/(1+V.length%2);
			float step = medianVector[inpIdx] - medianVector[inpIdx-skipNr];
			step /= skipNr;
			for (int i = 1; i < skipNr; i++)
			{
				medianVector[inpIdx-skipNr + i] = medianVector[inpIdx-skipNr] + i*step;

			}

			idx+=skipNr;
			inpIdx+=skipNr;
		}
		
		/*
		 * rewrite last part, errors!
		 * 
		 */

		inpIdx -= skipNr;

		if (inpIdx != Vector.length-1) // if the last entry was not covered in the loop above (frame number evenly divided by skipNr).
		{
//			V = removeEntry(V,Vector[Vector.length-W*skipNr]);
			V = removeEntry(V,Vector[idx-(2*W+1)*skipNr]);
			medianVector[medianVector.length-1] = (V[V.length/2] + (V.length%2)*V[V.length/2 + 1])/(1+V.length%2);
			float step = medianVector[medianVector.length-1] - medianVector[inpIdx];
			step /= (medianVector.length-inpIdx);
			for (int i = 1; i < medianVector.length-inpIdx; i++)
			{
				medianVector[inpIdx + i] = medianVector[inpIdx] + i*step;

			}
		}
		for (int i = 0; i < Vector.length; i++)
		{
			medianVector[i] = Vector[i]-medianVector[i];
			if (medianVector[i] < 0)
				medianVector[i] = 0;
		}
		return medianVector;

	}
	/*
	 * float precision version.
	 */
	public static float[] runningMedian(float[] Vector, int W){
		// Preallocate variables.				
		float[] medianVector = new float[Vector.length]; // Output vector.
		float[] V = new float[2*W+1];  // Vector for calculating running median.		

		int low = 0;
		int high = W;			   

		for(int i = 0; i < V.length; i++){ // Transfer first 2xW+1 entries.
			V[i] = Vector[i];
			if (i> W)
			{					
				quickSort(V, low, high); // Quicksort first W entries.	
				if (i % 2 == 0){
					medianVector[i-W-1] = (float) ((V[i/2]+V[i/2+1])/2.0);
				}else{
					medianVector[i-W-1] = V[i/2];

				}	
				high++;
			}			
		}


		for(int i = W; i < Vector.length-W-1; i++){ // Main loop, middle section.			
			medianVector[i] = V[W]; // Pull out median value.
			replaceEntry(V,Vector[i-W],Vector[i+W+1]);
			/*V = removeEntry(V,Vector[i-W]);
			V = sortInsert(V,Vector[i+W+1]);	*/

		}
		for (int i = Vector.length-W-1; i < Vector.length; i++){ // Last section, without access to data on right.			
			if (i % 2 == 0){				
				medianVector[i] = V[W-(i-Vector.length + W + 1)/2];
			}else{
				medianVector[i] = (float) ((V[W-(i-Vector.length + W)/2]+V[W-(i-Vector.length + W)/2-1])/2.0);
			}
			V = removeEntry(V,Vector[i-W]); // Remove items from V once per loop, ending with a W+1 large vector.	
		}		

		for (int i = 0; i < Vector.length; i++)
		{
			medianVector[i] = Vector[i]-medianVector[i];
			if (medianVector[i] < 0)
				medianVector[i] = 0;
		}
		return medianVector;
		//	return Vector;
	}

	public static void replaceEntry(float[] vector, float oldEntry, float newEntry)
	{
		int index = 0;
		boolean searching = true;
		while (searching)
		{
			if (vector[index]== oldEntry)
			{
				searching = false;
			}else
				index++;

			if (index == vector.length)
				searching = false;
		}		
		vector[index] = newEntry; // replace entry.
		quickSort(vector,0,vector.length-1);


	}
	public static float[] removeEntry(float[] inVector, float entry) { // Return vector with element "entry" missing.
		int found = 0;
		float[] vectorOut = new float[inVector.length -1];
		for (int i = 0; i < inVector.length - 1;i++){
			if (inVector[i] == entry){
				found = 1;				
			}
			vectorOut[i] = inVector[i+found];
		}
		return vectorOut;
	} 

	public static int indexOfIntArray(float[] array, float key) {
		int returnvalue = -1;
		for (int i = 0; i < array.length; ++i) {
			if (key == array[i]) {
				returnvalue = i;
				break;
			}
		}
		return returnvalue;
	}

	public static float[] sortInsert(float[] Vector, float InsVal){ // Assumes sorted input vector.
		float[] bigVector = new float[Vector.length + 1]; // Add InsVal into this vector		
		int indexToInsert = 0;
		if (InsVal > Vector[Vector.length-1]){ // If the value to be inserted is larger then the last value in the vector.

			System.arraycopy(Vector, 0, bigVector, 0, Vector.length);
			bigVector[bigVector.length-1] = InsVal;
			return bigVector;
		}else if (InsVal < Vector[0]){  // If the value to be inserted is smaller then the first value in the vector.
			bigVector[0] = InsVal;
			System.arraycopy(Vector, 0, bigVector, 1, Vector.length);
			return bigVector;
		}else{
			for (int i = 1; i < Vector.length; i++){
				if (InsVal < Vector[i]){
					indexToInsert = i;

					System.arraycopy(Vector, 0, bigVector, 0, indexToInsert);
					bigVector[indexToInsert] = InsVal;
					System.arraycopy(Vector, indexToInsert, bigVector, indexToInsert+1, Vector.length-indexToInsert);
					return bigVector;
				}

			}
		}
		return bigVector;
	}

	public static void quickSort(float[] arr, int low, int high) {
		if (arr == null || arr.length == 0)
			return;

		if (low >= high)
			return;

		// pick the pivot
		int middle = low + (high - low) / 2;
		float pivot = arr[middle];

		// make left < pivot and right > pivot
		int i = low, j = high;
		while (i <= j) {
			while (arr[i] < pivot) {
				i++;
			}

			while (arr[j] > pivot) {
				j--;
			}

			if (i <= j) {
				float temp = arr[i];
				arr[i] = arr[j];
				arr[j] = temp;
				i++;
				j--;
			}
		}

		// recursively sort two sub parts
		if (low < j)
			quickSort(arr, low, j);

		if (high > i)
			quickSort(arr, i, high);
	}
	/*
	public static double[] removeEntry(double[] inVector, double entry) { // Return vector with element "entry" missing.
		int found = 0;
		double[] vectorOut = new double[inVector.length -1];
		for (int i = 0; i < inVector.length - 1;i++){
			if (inVector[i] == entry){
				found = 1;				
			}
			vectorOut[i] = inVector[i+found];
		}
		return vectorOut;
	} 

	public static int indexOfIntArray(double[] array, double key) {
		int returnvalue = -1;
		for (int i = 0; i < array.length; ++i) {
			if (key == array[i]) {
				returnvalue = i;
				break;
			}
		}
		return returnvalue;
	}

	public static double[] sortInsert(double[] Vector, double InsVal){ // Assumes sorted input vector.
		double[] bigVector = new double[Vector.length + 1]; // Add InsVal into this vector		
		int indexToInsert = 0;
		if (InsVal > Vector[Vector.length-1]){ // If the value to be inserted is larger then the last value in the vector.

			System.arraycopy(Vector, 0, bigVector, 0, Vector.length);
			bigVector[bigVector.length-1] = InsVal;
			return bigVector;
		}else if (InsVal < Vector[0]){  // If the value to be inserted is smaller then the first value in the vector.
			bigVector[0] = InsVal;
			System.arraycopy(Vector, 0, bigVector, 1, Vector.length);

			return bigVector;
		}else{
			for (int i = 1; i < Vector.length; i++){
				if (InsVal < Vector[i]){
					indexToInsert = i;

					System.arraycopy(Vector, 0, bigVector, 0, indexToInsert);
					bigVector[indexToInsert] = InsVal;
					System.arraycopy(Vector, indexToInsert, bigVector, indexToInsert+1, Vector.length-indexToInsert);
					return bigVector;
				}

			}
		}
		return bigVector;
	}

	public static void quickSort(double[] arr, int low, int high) {
		if (arr == null || arr.length == 0)
			return;

		if (low >= high)
			return;

		// pick the pivot
		int middle = low + (high - low) / 2;
		double pivot = arr[middle];

		// make left < pivot and right > pivot
		int i = low, j = high;
		while (i <= j) {
			while (arr[i] < pivot) {
				i++;
			}

			while (arr[j] > pivot) {
				j--;
			}

			if (i <= j) {
				double temp = arr[i];
				arr[i] = arr[j];
				arr[j] = temp;
				i++;
				j--;
			}
		}

		// recursively sort two sub parts
		if (low < j)
			quickSort(arr, low, j);

		if (high > i)
			quickSort(arr, i, high);
	}*/
}