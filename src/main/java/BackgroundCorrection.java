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
import static jcuda.driver.JCudaDriver.cuCtxCreate;
import static jcuda.driver.JCudaDriver.cuCtxSynchronize;
import static jcuda.driver.JCudaDriver.cuDeviceGet;
import static jcuda.driver.JCudaDriver.cuInit;
import static jcuda.driver.JCudaDriver.cuLaunchKernel;
import static jcuda.driver.JCudaDriver.cuMemFree;
import static jcuda.driver.JCudaDriver.cuMemcpyDtoH;
import static jcuda.driver.JCudaDriver.cuModuleGetFunction;
import static jcuda.driver.JCudaDriver.cuModuleLoad;

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
				int stepLength = nFrames/30;
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
				cuModuleLoad(module, "medianFilter.ptx");
				// Obtain a handle to the kernel function.
				CUfunction function = new CUfunction();
				cuModuleGetFunction(function, module, "medianKernel");
				long GB = 1024*1024*1024;
				int frameSize = (3*columns*rows)*Sizeof.FLOAT;

				for(int Ch = 1; Ch <= nChannels; Ch++)
				{
					int staticMemory = (2*W[Ch-1]+1)*rows*columns*Sizeof.FLOAT;
					long framesPerBatch = (2*GB-staticMemory)/frameSize; // 3 GB memory allocation gives this numbers of frames. 					
				//	if (framesPerBatch > 5000)
				//		framesPerBatch = 5000;
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
						int stepLength = nFrames/30;
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
								Pointer.to(deviceOutput),
								Pointer.to(new int[]{testDataLength})
								);
						
//						Pointer.to(new int[]{stepLength}),
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

						// Free up memory allocation on device, housekeeping.

						cuMemFree(device_Data);    
						cuMemFree(deviceOutput);
						cuMemFree(device_meanVector);
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
								IP.set(i, hostOutput[idx]);
								idx++;
							}

							image.setProcessor(IP);
						} // frame loop for data return.

						startFrame = endFrame-W[Ch-1]; // include W more frames to ensure that border errors from median calculations dont occur ore often then needed.
						endFrame += framesPerBatch;					
						if (endFrame > nFrames)
							endFrame = nFrames;
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
		double[] temp = {1.0319345, 1.5788965, 1.5386362, 1.5084009, 1.3751887, 1.5217705, 1.1538985, 1.2564183, 1.0847589, 1.0095142, 0.95912206, 1.0614461, 1.0700893, 0.89172083, 0.8870745, 1.1037514, 1.0076519, 0.776869, 0.88048345, 1.0512484, 0.8226684, 0.8592002, 0.8469569, 0.82699686, 0.79158586, 0.9561204, 0.7326866, 0.8271085, 0.88200164, 0.830873, 1.0123901, 0.8907875, 0.96212, 1.0142025, 1.0313882, 0.9769623, 0.9488367, 0.9887663, 1.0230699, 1.0907819, 0.949085, 1.0025798, 0.8591082, 0.9437139, 0.9981515, 1.046564, 1.1354276, 1.1470896, 0.91952735, 1.2084074, 1.3454322, 1.2052082, 1.4043374, 1.5078149, 1.3296714, 1.0933654, 1.0207436, 1.0034858, 0.98945194, 0.98959285, 0.87834877, 0.8299667, 0.8942139, 1.29046, 0.89366245, 1.0662208, 1.3035733, 1.1207057, 1.192242, 1.3617904, 1.0735261, 1.0450656, 1.0001106, 0.9185965, 0.8652378, 1.0342845, 1.0342611, 1.0613121, 0.84495735, 1.0828689, 1.0282589, 1.0448864, 1.2458196, 1.2023321, 1.3349646, 1.3041848, 1.3407276, 1.3131496, 1.359266, 1.141944, 1.3194219, 1.177296, 1.070766, 1.1262398, 1.2937936, 1.2837096, 1.1297427, 1.0710608, 1.0944407, 1.0562905, 0.96722996, 1.149024, 1.2160307, 1.2390974, 1.0528122, 1.0297163, 1.132593, 1.0426215, 0.98812324, 0.9245492, 1.0785313, 1.0924127, 0.7990953, 0.86181146, 0.9583567, 0.84929866, 0.9688069, 1.098646, 1.0333395, 1.1743828, 1.2152102, 1.0006238, 1.0251718, 1.0787792, 1.109095, 0.9965485, 0.883124, 0.92979693, 0.9171921, 0.9886805, 1.0049182, 0.988979, 0.9975932, 1.0348837, 1.045568, 0.9643207, 0.9846807, 1.1963158, 1.2061926, 1.5583421, 1.5864364, 1.3865683, 1.1117823, 1.0549512, 1.2642275, 1.2776072, 1.3832319, 1.3666219, 1.0516979, 0.876722, 0.9062857, 1.0206703, 0.8997605, 0.8699513, 1.048114, 0.8885196, 0.98471093, 1.0339909, 0.99210495, 1.0633855, 1.2509391, 1.1131858, 1.2578546, 1.2781026, 1.249494, 1.2762727, 1.1016278, 1.199236, 1.2892741, 1.3120555, 1.1676912, 1.1710066, 1.0786434, 1.1625248, 1.1921566, 1.329776, 1.4409136, 1.2076113, 1.4754931, 1.3586311, 1.1065296, 1.0431701, 1.0897961, 1.1899356, 1.3749462, 1.2053173, 1.1159286, 1.2530469, 1.3007278, 1.0563927, 0.8917509, 1.1166652, 1.2543249, 1.4028398, 1.5498691, 1.4372042, 1.5015857, 1.2267575, 1.3150573, 1.6603433, 1.6142772, 1.3852924, 1.5539595, 1.2578403, 1.1427019, 1.4382952, 1.4072857, 1.3882124, 1.4576787, 1.3803581, 1.5113837, 1.2767183, 1.0857115, 1.0319608, 1.0412375, 0.99399173, 1.1976992, 1.4142386, 1.1494384, 1.1422225, 1.0447816, 1.0807586, 0.8700564, 1.0294831, 1.0965672, 1.0492444, 1.1737242, 1.2112426, 0.88977575, 0.90187943, 1.0803039, 0.9545235, 0.8011903, 1.2381325, 1.3615788, 1.456607, 1.2048271, 1.0749596, 1.0188292, 0.8084234, 0.94265413, 0.9723361, 1.0878638, 1.0576105, 1.2252063, 1.1007638, 0.8381269, 1.0302243, 1.1667962, 0.83095515, 0.8903823, 0.9783998, 1.0617685, 0.8138287, 0.9221841, 1.0005862, 0.95681477, 1.0035781, 0.8517374, 0.91693485, 0.87899774, 0.9575667, 0.9259708, 1.1965835, 0.88520354, 1.0830956, 0.983199, 0.98772675, 0.85053784, 1.101427, 0.91630435, 0.9828115, 1.0310187, 0.9580062, 1.0340528, 1.0063522, 1.1046654, 1.1148151, 1.0490806, 1.0610211, 1.2055756, 1.0871505, 0.9510593, 1.1439794, 1.426672, 1.3370335, 1.4181141, 1.073547, 1.0732123, 0.94313353, 0.90582156, 0.8810676, 0.99717516, 1.052502, 0.94669676, 0.9370489, 1.0023595, 0.9581481, 0.90442944, 0.8980585, 0.83523995, 1.0518936, 0.9512464, 0.88306874, 0.86697406, 0.9479267, 0.9912975, 0.93943024, 0.8742588, 0.8805644, 0.91538674, 0.92869276, 1.0144975, 1.0066458, 1.0395509, 0.93870986, 0.8939912, 1.1428324, 1.2099625, 1.5437843, 1.3632463, 1.3698608, 1.4264268, 1.2635047, 1.1724198, 1.0067438, 0.97613055, 1.0446465, 1.1280957, 1.0122466, 1.0655204, 0.9445591, 0.8747736, 1.0198412, 1.1741512, 0.9524121, 0.91843534, 0.96870524, 0.81209195, 1.0541083, 1.2157156, 1.3375204, 1.2857721, 1.6502552, 1.6883249, 1.2822053, 1.3328815, 1.5096484, 1.7033111, 1.2687843, 1.1135958, 1.3325187, 1.2267799, 1.4507085, 1.2494831, 0.9279691, 0.95828074, 0.9223457, 1.0291378, 1.0162479, 1.2111249, 1.1188956, 1.1570132, 1.0125479, 0.9483183, 0.9584979, 0.9108578, 0.9475869, 0.9569346, 1.1092384, 0.9881033, 0.88568, 0.8485529, 1.0230979, 0.95001805, 1.0251722, 0.9323073, 0.82065237, 0.8548272, 0.8868954, 1.0446851, 0.8941894, 0.9159207, 0.95704037, 0.9103063, 0.9197836, 0.9003321, 0.8669246, 0.86601573, 0.98155755, 1.0817301, 1.1465855, 1.008657, 1.0526131, 1.0879279, 0.9987189, 0.9232556, 1.1724243, 1.062351, 1.0349414, 1.2272484, 1.1564269, 0.93509364, 1.1081778, 1.0819548, 1.1256697, 1.0150927, 0.926655, 0.9674122, 0.8529948, 0.93121415, 0.8040616, 0.819774, 0.94768906, 0.89409274, 0.88469553, 1.1412421, 1.2246038, 0.88531286, 0.8960454, 0.84399176, 0.81527424, 0.8461738, 0.7762423, 0.9695537, 1.3696339, 1.3588209, 1.2426153, 1.13339, 1.0513936, 0.9939396, 0.8779321, 1.0318681, 0.9813628, 0.9520477, 0.97711694, 1.1563467, 1.1331728, 0.94585985, 1.081299, 1.1741841, 1.0732696, 1.0284157, 1.0701753, 1.4461036, 1.4423018, 1.4615656, 0.95403874, 1.0099151, 1.258667, 1.2851572, 1.3167555, 1.2575117, 1.055331, 1.0538733, 1.2337179, 1.3527361, 1.2106674, 0.8855018, 0.94171757, 0.80787903, 0.9025161, 0.9417427, 0.8973892, 0.90443397, 1.0777476, 0.9309698, 0.92111385, 1.1834488, 1.1948824, 1.0741462, 1.372846, 1.6280884, 1.4731989, 1.2431787, 1.0156, 1.3431659, 1.042329, 0.916523, 0.87167925, 0.89673966, 1.108916, 0.97777975, 1.060138, 1.0500566, 1.1454353, 1.202491, 1.1111823, 0.87790644, 1.1220376, 1.0120584, 1.029258, 0.87451935, 1.2250959, 0.8250646, 1.0926313, 1.1065731, 1.0094299, 0.79721725, 1.0058557, 0.990749, 1.0111917, 1.0622445, 0.917758, 0.9930846, 0.9990029, 1.0182279, 0.95865107, 0.9667441, 1.0246552, 0.96518534, 1.0177468, 0.9761925, 1.0338138, 1.0838486, 0.93554205, 0.9545771, 0.90329385, 1.0852088, 0.8737468, 0.9722748, 1.0547045, 1.005221, 0.99633974, 1.2868422, 1.3075827, 1.0946934, 0.9853768, 0.9697276, 0.9749025, 0.8650871, 0.8379979, 0.8991527, 1.008349, 1.0948591, 0.93315315, 1.1974938, 1.2044654, 1.067438, 0.9398203, 0.96615124, 0.82790333, 0.941419, 0.889042, 0.89419353, 0.85083884, 0.84644103, 0.93970877, 0.8647416, 0.9828093, 0.97312105, 1.2374603, 1.2204336, 1.0822654, 1.0514785, 0.9501328, 1.0451373, 0.98527, 1.0040625, 0.99226916, 1.4739184, 0.8241053, 1.078815, 0.9698602, 1.0236964, 1.0374672, 1.1309185, 1.1185011, 1.1747549, 1.1431127, 1.1317385, 1.0872413, 1.0647842, 1.0054014, 0.95373553, 0.91880214, 1.0760477, 0.96571314, 0.88784164, 1.3172432, 1.1902335, 1.2308074, 1.1153888, 1.1365584, 1.2156703, 1.1939851, 1.1040251, 1.309608, 1.3716301, 1.3418938, 1.075202, 1.005328, 1.0668523, 1.0305461, 0.9729648, 1.278405, 1.192258, 1.289675, 1.3762391, 1.334231, 1.1313937, 0.9525688, 1.1166261, 1.3335127, 0.980016, 0.8538339, 0.9977988, 0.8314463, 1.0151988, 0.8756405, 0.9132348, 0.81967586, 0.921984, 1.0072912, 0.98449117, 0.9091443, 0.8672228, 0.98894334, 0.8452709, 0.97067314, 1.0123461, 0.93643355, 1.0900384, 1.0600947, 0.91143906, 0.9373426, 1.0317668, 1.1633971, 1.0035529, 1.1107867, 1.2759212, 1.1242985, 1.2186673, 1.2742298, 1.3909228, 0.9358398, 0.85340667, 1.0571612, 0.8766589, 0.9345923, 0.86871415, 0.9864619, 0.834205, 0.9030317, 1.0288072, 0.9593543, 0.8578644, 0.925672, 0.8487444, 0.90247416, 0.88435024, 0.97206783, 1.0671623, 1.0153651, 0.9618235, 0.87697667, 0.8752317, 0.82008654, 0.7509405, 0.8227164, 1.0523509, 1.0543921, 1.0554668, 1.1089705, 0.99460226, 0.8375102, 1.0135466, 0.99486244, 0.9269758, 1.1123471, 1.0405595, 0.8455576, 0.92468226, 0.9344796, 1.0516042, 1.4652911, 1.4746112, 1.3617156, 1.2779276, 1.1872091, 1.2774369, 1.2785875, 1.1875211, 1.0076183, 1.0121374, 1.0502597, 1.0946087, 0.99511963, 0.8936587, 0.8669651, 0.8476798, 1.0790823, 1.0038842, 1.1467499, 0.97728276, 1.1765988, 1.370723, 1.3760822, 0.8492769, 0.94078565, 1.0576037, 1.0536677, 0.9481869, 1.1435359, 1.1532032, 1.1312518, 1.1960256, 1.1442287, 1.0936277, 1.0369943, 1.3159343, 1.1364752, 0.8949694, 0.8132914, 1.074676, 0.9554612, 0.9819524, 0.8112697, 1.0341159, 1.257867, 1.0879712, 1.2258232, 1.0784439, 1.0406545, 1.0330003, 1.060215, 0.94010997, 0.9864436, 1.2943009, 1.1979907, 0.9219469, 1.0434386, 0.8198764, 1.0171189, 1.27498, 1.3580146, 1.3129159, 1.3076506, 1.5037972, 1.6538808, 1.7061238, 1.5433453, 1.3194841, 1.2158953, 1.130451, 1.3499327, 1.1442543, 1.2149392, 1.3178102, 1.4402801, 1.2757907, 1.3322031, 1.4516984, 1.266225, 1.2647936, 1.1879455, 1.403231, 1.3513534, 1.1854228, 1.385336, 1.2457842, 1.255308, 1.3536439, 1.7172759, 1.6307697, 1.6341708, 1.7839204, 1.7951262, 1.5531113, 1.280387, 1.2767332, 1.0768596, 1.0615823, 1.0002911, 1.2630719, 0.92546743, 0.9646624, 0.98690516, 0.9996849, 0.91907305, 1.3467064, 1.2564207, 1.0478607, 0.800825, 1.0594716, 0.82655793, 0.9174155, 0.9341121, 0.8790726, 0.9208385, 1.0184507, 0.97785693, 0.99533296, 0.99995804, 0.9883214, 1.0082757, 0.8460255, 0.9167819, 0.98377824, 1.1032752, 0.9651501, 0.85605776, 0.9960022, 0.899194, 1.0195003, 0.86274916, 0.93404776, 1.0641081, 0.9804529, 1.0485197, 0.8237722, 1.0671835, 0.99020135, 0.8617063, 0.9850411, 0.9694626, 1.0700806, 0.9639042, 1.1903368, 1.2741414, 1.3220603, 1.1501046, 1.1185192, 1.1712976, 0.92311454, 0.9111483, 0.97856635, 0.9000859, 0.9751221, 1.225774, 1.0111558, 0.9314546, 0.9468853, 0.91072667, 0.9093358, 1.0904236, 1.0470866, 1.0360479, 0.971785, 0.90930384, 0.9331215, 1.1394842, 1.006303, 1.0211096, 1.0816994, 0.9912484, 0.9553417, 1.0443267, 0.8289404, 0.7315983, 0.878169, 0.89142466, 0.8345793, 0.81383985, 0.78423107, 0.8814389, 0.9096804, 0.82777613, 0.91255766, 0.99004215, 1.2314377, 0.8878097, 1.0613325, 1.0718542, 0.90912944, 1.1139586, 1.3047057, 1.1742084, 1.2989185, 1.197185, 1.1481737, 0.9654716, 0.8834164, 1.031658, 1.0492067, 1.0826907, 0.9441083, 0.91704774, 0.89401644, 0.99231225, 0.8845669, 0.8924465, 0.9553682, 0.87929195, 0.92629564, 0.82655025, 0.91598535, 0.8758587, 0.92430097, 0.8944933, 1.019845, 0.8916948, 0.91539896, 0.9533633, 0.72015697, 0.757752, 0.84344095, 0.862983, 0.76198316, 0.7440969, 0.9735196, 0.81786364, 0.9567371, 0.8637756, 0.84777623, 0.89311075, 0.7212245, 0.7156318, 0.9815745, 0.89724445, 0.8688192, 0.9150948, 0.96028805, 0.87717813, 0.829173, 0.80410457, 1.0711937, 1.1767106, 1.0416442, 1.124668, 1.147257, 1.0644133, 1.034725, 0.8500603, 0.8006272, 0.9338464, 0.79024345, 0.94666904, 0.8848174, 0.78495824, 0.9413045, 0.9400167, 1.1138846, 1.3226172, 1.1618114, 1.1334981, 1.076926, 1.0112759, 0.95335615, 1.1168319, 1.0786889, 1.0658649, 1.14869, 1.0184457, 0.8665059, 1.0736139, 1.0690191, 0.88457364, 0.8215275, 0.8685383, 0.93091255, 0.88598895, 0.81422466, 0.9281404, 0.8952881, 0.76347584, 0.85791075, 0.82913035, 0.8266119, 0.827182, 0.9846069, 0.935468, 0.90350974, 0.80096394, 0.7587798, 0.83607084, 0.8226184, 0.94755536, 0.8868788, 1.1306573, 1.152127, 1.1769278, 0.9534541, 0.92351943, 1.00033, 0.9414991, 0.86927694, 0.9372556, 0.91055095, 0.9175773, 0.8586348, 0.9283617, 0.7986044, 1.1273713, 1.0115399, 0.89832133, 1.3586078, 1.0921785, 0.9403217, 0.90843505, 0.95735353, 1.1440417, 1.116991, 0.98861253, 1.2123617, 1.2675672, 1.2288132, 1.0940671, 0.8675377, 0.8170163, 0.94584644};
		float[] data = new float[temp.length];
		for(int i = 0; i < data.length; i++)
			data[i] = (float) temp[i];		
		long time = System.nanoTime();
		for(int i = 0; i < 10; i ++)
			runningMedian(data,50);
		long time2 = System.nanoTime();		
		for(int i = 0; i < 10; i ++)
			runningMedian(data,50,4);
		long time3 = System.nanoTime();
		System.out.println("old:" + (time2-time)*1E-7 + " vs " + (time3 - time2)*1E-7 + " for total: "+ (float)((time2-time)/(time3-time2)));


		float[] old = runningMedian(data,50);
		float[] New = runningMedian(data,50,5);
		double[] error = new double[old.length];



		for (int i = 0; i < old.length; i++)
			error[i]  = (100*(New[i] - old[i]));

		correctDrift.plot(error);

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

		while (idx < Vector.length)
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

		while (inpIdx < Vector.length)
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
		inpIdx -= skipNr;
		if (inpIdx != Vector.length-1)
		{
			V = removeEntry(V,Vector[Vector.length-W*skipNr]);
			medianVector[medianVector.length-1] = (V[V.length/2] + (V.length%2)*V[V.length/2 + 1])/(1+V.length%2);
			float step = medianVector[medianVector.length-1] - medianVector[inpIdx-skipNr];
			step /= (medianVector.length-inpIdx);
			for (int i = 1; i < medianVector.length-1-(inpIdx-skipNr); i++)
			{
				medianVector[inpIdx-skipNr + i] = medianVector[inpIdx-skipNr] + i*step;

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