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
import jcuda.runtime.JCuda;
import ij.IJ;
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

		if(selectedModel == 3) // parallel, taking X pixels per round to handle large inputs (full Z)
		{			
			int fMax = (int) ((IJ.maxMemory()-nChannels*columns*rows*nFrames*4 - nFrames*4)/(nFrames*4)); // number of pixels that can be entered at the same time using avaible memory.
			for (int Ch = 1; Ch <= nChannels; Ch++){ // Loop over all channels.
				// loop from here over nMax frame segments. 
				float[] MeanFrame = new float[nFrames]; 		// Will include frame mean value.
				
				ImageProcessor IP = image.getProcessor();		// get image processor for the stack.´
				int startPixel = 0;
				int endPixel = fMax/2;
				if (endPixel > columns*rows)
					endPixel = columns*rows;
				// calculate steplength for median filtering:
				int stepLength = nFrames/300;
				if (stepLength > 10)
					stepLength = 10;
				if(nFrames < 500)
					stepLength = 1;
				while (endPixel <= columns*rows)
				{
					float[][] timeVector = new float[endPixel-startPixel][nFrames];
					for (int Frame = 1; Frame < nFrames+1; Frame++)
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
						IP = image.getProcessor(); 			// Update processor to next slice.

						if (startPixel == 0)
						{
							for (int i = 0; i < columns*rows; i++)
							{
								MeanFrame[Frame-1] += IP.get(i);
							}
							MeanFrame[Frame-1] /= rows*columns;

							if (MeanFrame[Frame-1] < 1)
								MeanFrame[Frame-1] = 1;
						}
						int j = 0;
						for (int i = startPixel; i < endPixel; i++)
						{							
							timeVector[j][Frame-1] = (float) (IP.get(i )/MeanFrame[Frame-1]); // load data. 						
							j++;
						}
					} // Data loading.
						List<Callable<float[]>> tasks = new ArrayList<Callable<float[]>>();	// Preallocate.
						if (stepLength == 1){
							for (int i = startPixel; i < endPixel; i++)
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
							for (int i = startPixel; i < endPixel; i++)
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
							int j = 0;
							for (int i = startPixel; i < startPixel+parallelCompute.size(); i++){							// Loop over and transfer results.
								try {
									float[] data = parallelCompute.get(i).get();									
									for (int k = 0; k < data.length; k++){	
										timeVector[j][k] = (int)(data[k]*MeanFrame[k]);								
										
									}			
								} catch (ExecutionException e) {
									e.printStackTrace();
								}
								j++;
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
							int j = 0;
							for (int i = startPixel; i < endPixel; i++)
							{
								value = (int)timeVector[j][Frame-1];											
								IP.set(i , value);	
								j++;
							}
							
						}
//						image.updateAndDraw();		
						System.out.println(startPixel + " through " + endPixel);
						
					startPixel = endPixel;
					endPixel += fMax/2;
					if (endPixel > columns*rows)
						endPixel = columns*rows;
					if (startPixel == columns*rows)
						endPixel += 100;
					System.out.println(startPixel + " through " + endPixel);
				}
							
				FilterKernel gs = new FilterKernel(); // generate kernel for frame filtering (b-spline)
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
					IP.setIntArray(gs.filter(IP.getIntArray())); // filter frame using gs kernel.
				}
				image.updateAndDraw();					
			} // Channel loop.

		}else
			if(selectedModel == 0) // parallel, taking X frames per round to handle large input
		{
			int nMax = (int) ((IJ.maxMemory()-(2*nChannels*nFrames*columns*rows*4))/(columns*rows*(4)*4 + 4)); // Size of a frame (4) and the required float overhead. This assumes that no other window is open.
			/*double imageSize = (2*nChannels*nFrames*columns*rows*4);//(1024*1024*1024);
			System.out.println(columns*rows);
			System.out.println(columns*rows*8);
			System.out.println(columns*rows*8*nFrames);
			System.out.println(columns*rows*8*nFrames*nChannels);
			
			int nMax2 = (int)(IJ.maxMemory()/(columns*rows*16));
			nMax2 = nMax2 - 2*nChannels*nFrames/4;
			System.out.println(nMax + " vs " + nMax2);*/
			for (int Ch = 1; Ch <= nChannels; Ch++){ // Loop over all channels.
				// loop from here over nMax frame segments. 
				int startFrame = 1;
				int endFrame = nMax/2;
				if (endFrame > nFrames)
					endFrame = nFrames;
				while (endFrame <= nFrames)
				{
					int numFrames = endFrame - startFrame + 1;
					float[] MeanFrame = new float[numFrames]; 		// Will include frame mean value.
					float[][] timeVector = new float[rows*columns][numFrames];
					ImageProcessor IP = image.getProcessor();		// get image processor for the stack.´
					int indexing = 0;
					for (int Frame = startFrame; Frame < endFrame+1; Frame++){			
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
							MeanFrame[indexing] += IP.get(i);
						}
						MeanFrame[indexing] /= rows*columns;

						if (MeanFrame[indexing] < 1)
							MeanFrame[indexing] = 1;
						for (int i = 0; i < rows*columns; i++)
						{
							timeVector[i][indexing] = (float) (IP.get(i )/MeanFrame[indexing]); // load data. 						
						}
						indexing++;
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
					FilterKernel gs = new FilterKernel(); // generate kernel for frame filtering (b-spline)
					
					int endEdgeHandling = 0;
					if (endFrame != nFrames)
						endEdgeHandling = W[Ch -1];
					int startEdgeHandling = 0;
					if (startFrame != 1)
						startEdgeHandling = W[Ch -1];
					indexing = startEdgeHandling;
					for (int Frame = startFrame+startEdgeHandling; Frame <= endFrame-endEdgeHandling; Frame++) // store data.
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
							value = (int)timeVector[i][indexing];											
							IP.set(i , value);	
						}
						IP.setIntArray(gs.filter(IP.getIntArray())); // filter frame using gs kernel.																
						indexing++;
					}
					startFrame = endFrame-2*W[Ch-1]; // include W more frames to ensure that border errors from median calculations dont occur ore often then needed.
					if (endFrame == nFrames)
					{
						endFrame += 100;
					}
					else
					{
						endFrame += nMax/2;					
						if (endFrame > nFrames)
							endFrame = nFrames;	
					}
				}
				image.updateAndDraw();					
			} // Channel loop.

		}else
			if(selectedModel == 1) // parallel, retain old functional copy.
		{
			for (int Ch = 1; Ch <= nChannels; Ch++){ // Loop over all channels.
				// loop from here over nMax frame segments. 
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
				FilterKernel gs = new FilterKernel(); // generate kernel for frame filtering (b-spline)
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
					IP.setIntArray(gs.filter(IP.getIntArray())); // filter frame using gs kernel.
				}
				image.updateAndDraw();					
			} // Channel loop.

		}else // end parallel.
			if(selectedModel == 2) // GPU.
			{					
//				int maxGrid = (int)(Math.log(CUDA.getGrid())/Math.log(2))+1;		
				int maxGrid = 31;
//				long GB = 1024*1024*1024;
				int frameSize = (4*columns*rows)*Sizeof.FLOAT;
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

				cuModuleLoadDataEx(module, Pointer.to(ptxFile), 
			            0, new int[0], Pointer.to(new int[0]));

				// Obtain a handle to the kernel function.
				CUfunction functionMedianFilter = new CUfunction();
				cuModuleGetFunction(functionMedianFilter, module, "medianKernel");
				CUmodule moduleBSpline = new CUmodule();				
				
				String ptxFileNameBspline = "filterImage.ptx";				
				byte ptxFileBspline[] = CUDA.loadData(ptxFileNameBspline);
				cuModuleLoadDataEx(moduleBSpline, Pointer.to(ptxFileBspline), 
	            0, new int[0], Pointer.to(new int[0]));				
				
				CUfunction functionBpline = new CUfunction();
				cuModuleGetFunction(functionBpline, moduleBSpline, "filterKernel");
				for(int Ch = 1; Ch <= nChannels; Ch++)
				{
					int staticMemory = 2*(2*W[Ch-1]+1)*rows*columns*Sizeof.FLOAT;
					long total[] = { 0 };
					long free[] = { 0 };
					JCuda.cudaMemGetInfo(free, total);
//					System.out.println("Total "+total[0]/GB+" free "+free[0]/GB);
					long maxMemoryGPU = (long) (0.5*free[0]); 
					long framesPerBatch = (maxMemoryGPU-staticMemory)/frameSize; // maxMemoryGPU GB memory allocation gives this numbers of frames. 					
					framesPerBatch /= 2;
					int loadedFrames = 0;
					int startFrame = 1;					
					int endFrame = (int)framesPerBatch;					
					if (endFrame > nFrames)
						endFrame = nFrames;

					
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
								timeVector[frameCounter + (endFrame-startFrame+1)*i] = (float)IP.get(i);			
								MeanFrame[frameCounter] += (float)(IP.get(i));
							}
							MeanFrame[frameCounter] /= (float)(columns*rows);
							loadedFrames++;
							frameCounter++;
						} // frame loop for mean calculations.
						
						int stepLength = nFrames/300;
						if (stepLength > 10)
							stepLength = 10;
						if(nFrames < 500)
							stepLength = 1;					
						int nData = rows * columns;
						int blockSize = 256;
						int gridSize = (nData + blockSize - 1)/blockSize;
						gridSize = (int) (Math.log(gridSize)/Math.log(2) + 2);
						if (gridSize > maxGrid)
							gridSize = (int)( Math.pow(2, maxGrid));
						else
							gridSize = (int)( Math.pow(2, gridSize));
						CUdeviceptr device_Data 		= CUDA.copyToDevice(timeVector);
						CUdeviceptr device_meanVector 	= CUDA.copyToDevice(MeanFrame);
						CUdeviceptr deviceOutput 		= CUDA.allocateOnDevice((int)(timeVector.length));
						CUdeviceptr device_window 		= CUDA.allocateOnDevice((float)((2 * W[Ch-1] + 1) * rows * columns)); // swap vector.
						int filterWindowLength 		= (2 * W[Ch-1] + 1) * rows * columns;
						int dataLength 				= timeVector.length;
						int meanVectorLength 		= MeanFrame.length;

					//	ij.IJ.log(Integer.toString(W[Ch]));
						Pointer kernelParametersMedianFilter 	= Pointer.to(   
								Pointer.to(new int[]{nData}),
								Pointer.to(new int[]{W[Ch]}),
								Pointer.to(device_window),
								Pointer.to(new int[]{filterWindowLength}),
								Pointer.to(new int[]{(meanVectorLength)}),
								Pointer.to(device_Data),
								Pointer.to(new int[]{dataLength}),
								Pointer.to(device_meanVector),
								Pointer.to(new int[]{meanVectorLength}),								
								Pointer.to(new int[]{stepLength}),
								Pointer.to(deviceOutput),
								Pointer.to(new int[]{dataLength})
								);
						cuLaunchKernel(functionMedianFilter,
								gridSize,  1, 1, 	// Grid dimension
								blockSize, 1, 1,  // Block dimension
								0, null,               		// Shared memory size and stream
								kernelParametersMedianFilter, null 		// Kernel- and extra parameters
								);
						cuCtxSynchronize();
						cuMemFree(device_window);
						cuMemFree(device_Data);  
						cuMemFree(device_meanVector);
					
						// Temporary solution to error found in medianFilter.ptx, run that part on CPU:
						
						
						
						// B-spline filter image:

						double[] filterKernel = { 0.0015257568383789054, 0.003661718759765626, 0.02868598630371093, 0.0036617187597656254, 0.0015257568383789054, 
                                0.003661718759765626, 0.008787890664062511, 0.06884453115234379, 0.00878789066406251, 0.003661718759765626, 
                                0.02868598630371093, 0.06884453115234379, 0.5393295900878906, 0.06884453115234378, 0.02868598630371093,
                                0.0036617187597656254, 0.00878789066406251, 0.06884453115234378, 0.008787890664062508, 0.0036617187597656254, 
                                0.0015257568383789054, 0.003661718759765626, 0.02868598630371093, 0.0036617187597656254, 0.0015257568383789054}; // 5x5 bicubic Bspline filter.
						
						int bSplineDataLength = timeVector.length;
						nData = timeVector.length/(columns*rows);
				
						
						CUdeviceptr deviceOutputBSpline 		= CUDA.allocateOnDevice(bSplineDataLength);
						CUdeviceptr deviceFilterKernel 			= CUDA.copyToDevice(filterKernel); // filter to applied to each pixel.				
				
						Pointer kernelParametersBspline 		= Pointer.to(   
								Pointer.to(new int[]{nData}),
								Pointer.to(deviceOutput),					// input data is output from medianFilter function call.
								Pointer.to(new int[]{bSplineDataLength}),	// length of vector
								Pointer.to(new int[]{(rows)}), 			// width
								Pointer.to(new int[]{(columns)}),				// height
								Pointer.to(deviceFilterKernel),				// Transfer filter kernel.
								Pointer.to(new int[]{(int)filterKernel.length}),								
								Pointer.to(new int[]{(int)(Math.sqrt(filterKernel.length))}), // width of filter kernel.
								Pointer.to(deviceOutputBSpline),			// result vector.
								Pointer.to(new int[]{bSplineDataLength})
								);

						blockSize = 256;
						gridSize = (nData + blockSize - 1)/blockSize;
						gridSize = (int) (Math.log(gridSize)/Math.log(2) + 1);
						if (gridSize > maxGrid)
							gridSize = (int)( Math.pow(2, maxGrid));
						else
							gridSize = (int)( Math.pow(2, gridSize));

						cuLaunchKernel(functionBpline,
								gridSize,  1, 1, 	// Grid dimension
								blockSize, 1, 1,  // Block dimension
								0, null,               		// Shared memory size and stream
								kernelParametersBspline, null 		// Kernel- and extra parameters
								);
						
					/*	gridSizeX = (int)Math.ceil(Math.sqrt(timeVector.length/(columns*rows))); 	// update.
						gridSizeY = gridSizeX;														// update.
						cuLaunchKernel(functionBpline,
								gridSizeX,  gridSizeY, 1, 	// Grid dimension
								blockSizeX, blockSizeY, 1,  // Block dimension
								0, null,               		// Shared memory size and stream
								kernelParametersBspline, null 		// Kernel- and extra parameters
								);*/
						cuCtxSynchronize();

						
						
						int hostOutput[] = new int[timeVector.length];
						cuMemcpyDtoH(Pointer.to(hostOutput), deviceOutputBSpline,
								bSplineDataLength * Sizeof.INT);
						// clean up.
						cuMemFree(deviceOutput); 
						cuMemFree(deviceOutputBSpline);    
						cuMemFree(deviceFilterKernel);   
						int idx = 0;
						if (endFrame == nFrames) // if we're on the last bin, include last segment.
							endFrame += W[Ch-1];
						if (startFrame != 1) // if we're not on the first set of bins, ignore the first segment.
						{
							startFrame += W[Ch-1];
							idx += W[Ch-1]*rows*columns;
						}
						for (int Frame = startFrame; Frame <= endFrame-W[Ch-1]; Frame++)
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
							int i = 0; 							
							while (i < rows*columns)
							{
								if (hostOutput[idx] > 0)
									IP.set(i, (int)hostOutput[idx]);
								else
									IP.set(i, 0);
								idx++;
								i++;
							}

							image.setProcessor(IP);
						} // frame loop for data return.
				
						startFrame = endFrame-2*W[Ch-1]; // include W more frames to ensure that border errors from median calculations dont occur ore often then needed.
						endFrame += framesPerBatch;					
						if (endFrame > nFrames)
							endFrame = nFrames;					
					} // while loadedChannels < nFrames
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
					medianVector[i-W-1] = (float) ((V[i/2]+V[i/2-1])/2.0);
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
				medianVector[i] = V[W-(i-Vector.length + W)/2];
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
}