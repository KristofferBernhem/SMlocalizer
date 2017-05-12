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
import static jcuda.driver.JCudaDriver.cuModuleLoadDataEx;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

import ij.ImagePlus;
import ij.WindowManager;
import ij.plugin.filter.Analyzer;
import ij.process.ImageProcessor;
import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.CUcontext;
import jcuda.driver.CUdevice;
import jcuda.driver.CUdeviceptr;
import jcuda.driver.CUfunction;
import jcuda.driver.CUmodule;


/*
 * TODO: Fiducial correction on fits, add call with logical array.
 */
public class localizeAndFit {

	public static ArrayList<Particle> run(int[] MinLevel, int gWindow, int inputPixelSize,  int[] totalGain , int selectedModel, double maxSigma, String modality){						
		ImagePlus image 					= WindowManager.getCurrentImage();
		int columns 						= image.getWidth();
		int rows 							= image.getHeight();

		int nChannels 						= image.getNChannels(); 	// Number of channels.
		int nFrames 						= image.getNFrames();
		if (nFrames == 1)
			nFrames 						= image.getNSlices();

		int minPosPixels = 0;
		if (modality.equals("2D"))
		{
			minPosPixels = gWindow*gWindow - 4; // update to relevant numbers for this modality.
		}
		else if (modality.equals("PRILM"))
		{
			minPosPixels = gWindow*gWindow - 4; // update to relevant numbers for this modality.
		}
		else if (modality.equals("Biplane"))
		{
			// get calibrated values for gWindow.
			minPosPixels = gWindow*gWindow - 4; // update to relevant numbers for this modality.
		}
		else if (modality.equals("Double Helix"))
		{
			// get calibrated values for gWindow.
			minPosPixels = gWindow*gWindow - 4; // update to relevant numbers for this modality.
		}
		else if (modality.equals("Astigmatism"))
		{
			// get calibrated values for gWindow.
			minPosPixels = gWindow*gWindow - 4; // update to relevant numbers for this modality.					
			if (minPosPixels > 50)
				minPosPixels = 50;
		}

		ArrayList<Particle> Results 		= new ArrayList<Particle>();		// Fitted results array list.
		if (selectedModel == 0) // parallel
		{
			ImageProcessor IP = image.getProcessor();
			ArrayList<fitParameters> fitThese 	= new ArrayList<fitParameters>(); 	// arraylist to hold all fitting parameters.

			for (int Ch = 1; Ch <= nChannels; Ch++)							// Loop over all channels.
			{
				int[] limits = findLimits.run(image, Ch);
				for (int Frame = 1; Frame <= nFrames;Frame++)					// Loop over all frames.
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
					int[][] Arr = IP.getIntArray();

					//					ArrayList<int[]> Center 	= LocalMaxima.FindMaxima(Arr, gWindow[Ch-1], MinLevel[Ch-1], sqDistance[Ch-1], minPosPixels[Ch-1]); 	// Get possibly relevant center coordinates.
					ArrayList<int[]> Center 	= LocalMaxima.FindMaxima(IP, gWindow, limits[Frame-1], minPosPixels); 	// Get possibly relevant center coordinates.
					
					/*
					 * loop over all frames and send calclulate minlevel based on 100 frame means.
					 */
					for (int i = 0; i < Center.size(); i++)
					{

						int[] dataFit = new int[gWindow*gWindow];			// Container for data to be fitted.
						int[] Coord = Center.get(i);									// X and Y coordinates for center pixels to be fitted.

						for (int j = 0; j < gWindow*gWindow; j++)
						{
							int x =  Coord[0] - Math.round((gWindow)/2) +  (j % gWindow);
							int y =  Coord[1] - Math.round((gWindow)/2) +  (j / gWindow);
							dataFit[j] = Arr[x][y];	// Faster then pulling from IP.get again.
						} // pull out data for this fit.

						fitThese.add(new fitParameters(Coord, 
								dataFit,
								Ch,
								Frame,
								inputPixelSize,
								gWindow,
								totalGain[Ch-1]));

					} // loop over all located centers from this frame.									
				} // loop over all frames.									
			} // loop over all channels.
			List<Callable<Particle>> tasks = new ArrayList<Callable<Particle>>();	// Preallocate.

			for (final fitParameters object : fitThese) {							// Loop over and setup computation.
				Callable<Particle> c = new Callable<Particle>() {					// Computation to be done.
					@Override
					public Particle call() throws Exception {
						return ParticleFitter.Fitter(object,maxSigma);						// Actual call for each parallel process.
					}
				};
				tasks.add(c);														// Que this task.
			} // setup parallel computing, distribute all fitThese objects. 

			int processors 			= Runtime.getRuntime().availableProcessors();	// Number of processor cores on this system.
			ExecutorService exec 	= Executors.newFixedThreadPool(processors);		// Set up parallel computing using all cores.
			try {

				List<Future<Particle>> parallelCompute = exec.invokeAll(tasks);				// Execute computation.    
				for (int i = 0; i < parallelCompute.size(); i++){							// Loop over and transfer results.
					try {
						Results.add(parallelCompute.get(i).get());							// Add computed results to Results arraylist.
					} catch (ExecutionException e) {
						e.printStackTrace();
					}
				}


			} catch (InterruptedException e) {

				e.printStackTrace();
			}
			finally {
				exec.shutdown();
			} // try executing parallel computation.


			//TableIO.Store(Results);												// Return and display results to user.
			ArrayList<Particle> cleanResults = new ArrayList<Particle>();
			/*
			 * Remove non-realistic fits.
			 */			
			for (int i = 0; i < Results.size(); i++)
			{
				if (Results.get(i).x > 0 &&
						Results.get(i).y > 0 &&						
						Results.get(i).sigma_x > 0 &&
						Results.get(i).sigma_y > 0 &&
						Results.get(i).precision_x > 0 &&
						Results.get(i).precision_y > 0 &&
						Results.get(i).photons > 0 && 
						Results.get(i).r_square > 0)
					cleanResults.add(Results.get(i));
			}

			/*
			 * Remove duplicates that can occur if two center pixels has the exact same value. Select the best fit if this is the case.
			 */
			int currFrame = -1;
			int i  = 0;
			int j  = 0;
			int Ch = 1;

			int pixelDistance = 2*inputPixelSize*inputPixelSize;
			while( i < cleanResults.size())
			{				
				if (cleanResults.get(i).channel > Ch)
					pixelDistance = 2*inputPixelSize*inputPixelSize;
				if( cleanResults.get(i).frame > currFrame)
				{
					currFrame = cleanResults.get(i).frame;					
					j = i+1;

				}
				while (j < cleanResults.size() && cleanResults.get(j).frame == currFrame)
				{
					if (((cleanResults.get(i).x - cleanResults.get(j).x)*(cleanResults.get(i).x - cleanResults.get(j).x) + 
							(cleanResults.get(i).y - cleanResults.get(j).y)*(cleanResults.get(i).y - cleanResults.get(j).y)) < pixelDistance)
					{
						if (cleanResults.get(i).r_square > cleanResults.get(j).r_square)
						{
							cleanResults.get(j).include = 0;					
						}else
						{
							cleanResults.get(i).include = 0;
						}
					}
					j++;
				}

				i++;
				j = i+1;
			}
			for (i = cleanResults.size()-1; i >= 0; i--)
			{
				if (cleanResults.get(i).include == 0)
					cleanResults.remove(i);
			}


			if (modality.equals("2D"))
			{
				cleanResults = BasicFittingCorrections.compensate(cleanResults); // change 2D data to 3D data based on calibration data.
			}
			else if (modality.equals("PRILM"))
			{
				cleanResults = PRILMfitting.fit(cleanResults); // change 2D data to 3D data based on calibration data.
			}
			else if (modality.equals("Biplane"))
			{
				cleanResults = BiplaneFitting.fit(cleanResults,inputPixelSize,totalGain); // change 2D data to 3D data based on calibration data.
				columns /= 2;
			}
			else if (modality.equals("Double Helix"))
			{
				cleanResults = DoubleHelixFitting.fit(cleanResults); // change 2D data to 3D data based on calibration data.
			}
			else if (modality.equals("Astigmatism"))
			{
				cleanResults = AstigmatismFitting.fit(cleanResults); // change 2D data to 3D data based on calibration data.	
			}
			ij.measure.ResultsTable tab = Analyzer.getResultsTable();
			tab.reset();		
			tab.incrementCounter();
			tab.addValue("width", columns*inputPixelSize);
			tab.addValue("height", rows*inputPixelSize);
			tab.show("Results");

			TableIO.Store(cleanResults);
			return cleanResults; // end parallel computation by returning results.
		}else // end parallel. 
			if (selectedModel == 2) // GPU
			{			
				double convCriteria = 1E-8; // how large improvement from one step to next we require.
				int maxIterations = 1000;  // stop if an individual fit reaches this number of iterations.
				ImageProcessor IP = image.getProcessor();
				
				// Initialize the driver and create a context for the first device.
				cuInit(0);
				CUdevice device = new CUdevice();
				cuDeviceGet(device, 0);
				CUcontext context = new CUcontext();
				cuCtxCreate(context, 0, device);

				// gauss fit algorithm.
				// Load the PTX that contains the kernel.
				CUmodule module = new CUmodule();
				//				cuModuleLoad(module, "gFit.ptx");
				String ptxFileNameGaussFit = "gFit.ptx";
				byte ptxFileGaussFit[] = CUDA.loadData(ptxFileNameGaussFit);
				//				cuModuleLoad(module, "medianFilter.ptx"); // old  version, loading directly from the ptx file.
				//cuModuleLoadDataEx(module,ptxFile);
				cuModuleLoadDataEx(module, Pointer.to(ptxFileGaussFit), 
						0, new int[0], Pointer.to(new int[0]));
				// Obtain a handle to the kernel function.
				CUfunction fittingFcn = new CUfunction();
				cuModuleGetFunction(fittingFcn, module, "gaussFitter");  //gFit.pth (gaussFitter function).

				// prepare for locating centra for gaussfitting.
				// Load the PTX that contains the kernel.
				CUmodule moduleLM = new CUmodule();
				String ptxFileNameFindMaxima = "findMaxima.ptx";
				byte ptxFileFindMaxima[] = CUDA.loadData(ptxFileNameFindMaxima);
				//	cuModuleLoad(moduleLM, "findMaxima.ptx");					
				// Obtain a handle to the kernel function
				cuModuleLoadDataEx(moduleLM, Pointer.to(ptxFileFindMaxima), 
						0, new int[0], Pointer.to(new int[0]));
				CUfunction findMaximaFcn = new CUfunction();
				cuModuleGetFunction(findMaximaFcn, moduleLM, "run");	// findMaxima.ptx (run function).
				//CUmodule modulePrepGauss = new CUmodule();
				//cuModuleLoad(modulePrepGauss, "prepareGaussian.ptx");					
				// Obtain a handle to the kernel function
				//	CUfunction prepareGaussFcn = new CUfunction();
				//	cuModuleGetFunction(prepareGaussFcn, modulePrepGauss, "run");	// prepareGaussian.ptx (run function).



				for (int Ch = 1; Ch <= nChannels; Ch++)					// Loop over all channels.
				{						
				//	int[] limits = findLimits.run(image, Ch); // get limits.
					int nCenter =(( columns*rows/(gWindow*gWindow)) / 2); // ~ 80 possible particles for a 64x64 frame. Lets the program scale with frame size.
					long gb = 1024*1024*1024;
					long maxMemoryGPU = 3*gb; // TODO: get size of gpu memory.
					int nMax = (int) (maxMemoryGPU/(2*columns*rows*Sizeof.INT + 4*nCenter*Sizeof.INT)); 	// the localMaxima GPU calculations require: (x*y*frame*(Sizeof.INT ) + frame*nCenters*Sizeof.FLOAT)/gb memory. with known x and y dimensions, determine maximum size of frame for each batch.
				//	if (nMax > 10000)
				//		nMax = 10000;
					//ArrayList<fitParameters> fitThese = new ArrayList<fitParameters>();
					int dataIdx = 0;
					int idx = 0;					
					double lowXY = gWindow/2 - 2;
					if (lowXY < 1)
						lowXY = 1;
					double highXY = gWindow/2 +  2;
					double[] bounds = { // bounds for gauss fitting.
							0.6			, 1.4,				// amplitude.
							lowXY	, highXY,			// x.
							lowXY	, highXY,			// y.
							0.8			,  maxSigma,		// sigma x.
							0.8			,  maxSigma,		// sigma y.
							(-0.5*Math.PI) , (0.5*Math.PI),	// theta.
							-0.5		, 0.5				// offset.
					};
					CUdeviceptr deviceBounds 		= CUDA.copyToDevice(bounds);															
					if (nMax > nFrames)
						nMax = nFrames;

					int[] data = new int[columns*rows*nMax];	
					boolean processed = true;
					int startFrame = 0;
					for (int Frame = 1; Frame <= nFrames; Frame ++)
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
						if(processed)
						{
							processed = false;
							startFrame = Frame;
						}


						// load as large chunks of data as possible on the gpu at each time.
						if (dataIdx < nMax)
						{
							for(int i = 0; i < columns*rows;i++) // loop over X then Y.
							{
								data[idx] = IP.get(i);
								idx++;
							}							
							dataIdx++; // count up number of added frames this round.		    			
						}
						if (dataIdx == nMax)
						{
							dataIdx = 0; // reset for next round.
							idx = 0; // reset for next round.		
							processed=true;
							CUdeviceptr deviceData 	= CUDA.copyToDevice(data);
							int[] limitsN = findLimits.run(data, columns, rows, Ch); // get limits.							
							CUdeviceptr deviceLimits 	= CUDA.copyToDevice(limitsN);
							CUdeviceptr deviceCenter = CUDA.allocateOnDevice((int)(nMax*nCenter));
							Pointer kernelParameters 		= Pointer.to(   
									Pointer.to(deviceData),
									Pointer.to(new int[]{data.length}),
									Pointer.to(new int[]{columns}),		 				       
									Pointer.to(new int[]{rows}),
									Pointer.to(new int[]{gWindow}),
									Pointer.to(deviceLimits),
									Pointer.to(new int[]{limitsN.length}),
									Pointer.to(new int[]{minPosPixels}),
									Pointer.to(new int[]{nCenter}),
									Pointer.to(deviceCenter),
									Pointer.to(new int[]{nMax*nCenter}));

							int blockSizeX 	= 1;
							int blockSizeY 	= 1;				   
							int gridSizeX 	= (int) Math.ceil((Math.sqrt(nMax)));
							int gridSizeY 	= gridSizeX;

							cuLaunchKernel(findMaximaFcn,
									gridSizeX,  gridSizeY, 1, 	// Grid dimension
									blockSizeX, blockSizeY, 1,  // Block dimension
									0, null,               		// Shared memory size and stream
									kernelParameters, null 		// Kernel- and extra parameters
									);
							cuCtxSynchronize();

							int hostCenter[] = new int[nMax*nCenter];
							// Pull data from device.
							cuMemcpyDtoH(Pointer.to(hostCenter), deviceCenter,
									nMax*nCenter * Sizeof.INT);

							// Free up memory allocation on device, housekeeping.
							cuMemFree(deviceCenter);
							cuMemFree(deviceData);
							cuMemFree(deviceLimits);
							/******************************************************************************
							 * Transfer data for gauss fitting.
							 ******************************************************************************/
							int newN = 0;
							for (int j = 0; j < hostCenter.length; j++) // loop over all possible centers.
							{
								if (hostCenter[j]> 0) // if the center was added.
								{									
									newN++;
								}
							}

							int loaded = 0;
							while (loaded < newN)
							{
								int maxLoad = 100000;
								if ((newN - loaded) < maxLoad)
									maxLoad = newN - loaded;
								int[] locatedCenter = new int[maxLoad]; // cleaned vector with indexes of centras.
								int[] locatedFrame = new int[maxLoad]; // cleaned vector with indexes of centras.
								int counter = 0;
								int j = loaded;
								boolean fill = true;
								while (fill)
								{
									if (hostCenter[j]> 0 )
									{
										locatedCenter[counter] = hostCenter[j];													
										locatedFrame[counter] = hostCenter[j]/(columns*rows);							
										counter++;
									}

									j ++;
									if (counter == maxLoad || j == hostCenter.length)
										fill = false;										
								}
								double[] P = new double[counter*7];
								double[] stepSize = new double[counter*7];							
								int[] gaussVector = new int[counter*gWindow*gWindow];

								for (int i = 0; i < counter; i++)
								{
									P[i*7] = data[locatedCenter[i]];
									P[i*7+1] = 2;
									P[i*7+2] = 2;
									P[i*7+3] = 1.5;
									P[i*7+4] = 1.5;
									P[i*7+6] = 0;
									P[i*7+6] = 0;
									stepSize[i * 7] = 0.1;// amplitude
									stepSize[i * 7 + 1] = 0.25*100/inputPixelSize; // x center.
									stepSize[i * 7 + 2] = 0.25*100/inputPixelSize; // y center.
									stepSize[i * 7 + 3] = 0.25*100/inputPixelSize; // sigma x.
									stepSize[i * 7 + 4] = 0.25*100/inputPixelSize; // sigma y.
									stepSize[i * 7 + 5] = 0.19625; // Theta.
									stepSize[i * 7 + 6] = 0.01; // offset.   
									int k = locatedCenter[i] - (gWindow / 2) * (columns + 1); // upper left corner.
									j = 0;
									int loopC = 0;
									while (k <= locatedCenter[i] + (gWindow / 2) * (columns + 1)) // loop over all relevant pixels. use this loop to extract data based on single indexing defined centers.
									{
										gaussVector[i * gWindow * gWindow + j] = data[k]; // add data.
										k++;
										loopC++;
										j++;
										if (loopC == gWindow)
										{
											k += (columns - gWindow);
											loopC = 0;
										}
									} // data pulled.
								}

								CUdeviceptr deviceGaussVector 	= 		CUDA.copyToDevice(gaussVector);					
								CUdeviceptr deviceP 			= 		CUDA.copyToDevice(P);
								CUdeviceptr deviceStepSize 		= 		CUDA.copyToDevice(stepSize);							

								/******************************************************************************
								 * Gauss fitting.
								 ******************************************************************************/

								gridSizeX = (int) Math.ceil((Math.sqrt(counter)));
								gridSizeY 	= gridSizeX;
								Pointer kernelParametersGaussFit 		= Pointer.to(   
										Pointer.to(deviceGaussVector),
										Pointer.to(new int[]{counter * gWindow * gWindow}),
										Pointer.to(deviceP),																											
										Pointer.to(new double[]{counter*7}),
										Pointer.to(new short[]{(short) gWindow}),
										Pointer.to(deviceBounds),
										Pointer.to(new double[]{bounds.length}),
										Pointer.to(deviceStepSize),																											
										Pointer.to(new double[]{counter*7}),
										Pointer.to(new double[]{convCriteria}),
										Pointer.to(new int[]{maxIterations}));	

								cuLaunchKernel(fittingFcn,
										gridSizeX,  gridSizeY, 1, 	// Grid dimension
										blockSizeX, blockSizeY, 1,  // Block dimension
										0, null,               		// Shared memory size and stream
										kernelParametersGaussFit, null 		// Kernel- and extra parameters
										);
								cuCtxSynchronize(); 

								double hostOutput[] = new double[counter*7];

								// Pull data from device.
								cuMemcpyDtoH(Pointer.to(hostOutput), deviceP,
										counter*7 * Sizeof.DOUBLE);
								// Free up memory allocation on device, housekeeping.
								cuMemFree(deviceGaussVector);   
								cuMemFree(deviceP);    
								cuMemFree(deviceStepSize);	
								for (int n = 0; n < counter; n++) //loop over all particles
								{	    	
									Particle Localized = new Particle();
									Localized.include 		= 1;
									Localized.channel 		= Ch;
									Localized.frame   		= startFrame + locatedFrame[n];//+ locatedCenter[n]/(columns*rows);
									Localized.r_square 		= hostOutput[n*7+6];
									Localized.x				= inputPixelSize*(0.5 + hostOutput[n*7+1] + (locatedCenter[n]%columns) - Math.round((gWindow)/2));
									Localized.y				= inputPixelSize*(0.5 + hostOutput[n*7+2] + ((locatedCenter[n]/columns)%rows) - Math.round((gWindow)/2));
									Localized.z				= inputPixelSize*0;	// no 3D information.
									Localized.sigma_x		= inputPixelSize*hostOutput[n*7+3];
									Localized.sigma_y		= inputPixelSize*hostOutput[n*7+4];
									Localized.photons		= (int) (hostOutput[n*7]/totalGain[Ch-1]);
									Localized.precision_x 	= Localized.sigma_x/Math.sqrt(Localized.photons);
									Localized.precision_y 	= Localized.sigma_y/Math.sqrt(Localized.photons);
									Results.add(Localized);
								}	
								loaded += maxLoad;

							}

						}else if ( Frame == nFrames) // final part if chunks were loaded.
						{
						
							int[] remainingData = new int[idx];
							for (int i = 0; i < idx; i++)
								remainingData[i] = data[i];

							while (idx < data.length)
							{
								data[idx] = 0; // remove remaining entries.
								idx++;
							}
							processed = true;
							CUdeviceptr deviceData 	= CUDA.copyToDevice(remainingData);							
							int[] limitsN = findLimits.run(data, columns, rows, Ch); // get limits.
							CUdeviceptr deviceLimits 	= CUDA.copyToDevice(limitsN);
							CUdeviceptr deviceCenter = CUDA.allocateOnDevice((int)(nMax*nCenter));
							Pointer kernelParameters 		= Pointer.to(   
									Pointer.to(deviceData),
									Pointer.to(new int[]{remainingData.length}),
									Pointer.to(new int[]{columns}),		 				       
									Pointer.to(new int[]{rows}),
									Pointer.to(new int[]{gWindow}),
									Pointer.to(deviceLimits),
									Pointer.to(new int[]{limitsN.length}),
									Pointer.to(new int[]{minPosPixels}),
									Pointer.to(new int[]{nCenter}),
									Pointer.to(deviceCenter),
									Pointer.to(new int[]{nMax*nCenter}));
							
						
							int blockSizeX 	= 1;
							int blockSizeY 	= 1;				   
							int gridSizeX 	= (int) Math.ceil((Math.sqrt(nMax)));
							int gridSizeY 	= gridSizeX;

							cuLaunchKernel(findMaximaFcn,
									gridSizeX,  gridSizeY, 1, 	// Grid dimension
									blockSizeX, blockSizeY, 1,  // Block dimension
									0, null,               		// Shared memory size and stream
									kernelParameters, null 		// Kernel- and extra parameters
									);
							cuCtxSynchronize();

							int hostCenter[] = new int[nMax*nCenter];

							// Pull data from device.
							cuMemcpyDtoH(Pointer.to(hostCenter), deviceCenter,
									nMax*nCenter * Sizeof.INT);

							// Free up memory allocation on device, housekeeping. 
							cuMemFree(deviceCenter);
							cuMemFree(deviceData);
							cuMemFree(deviceLimits);
							/******************************************************************************
							 * Transfer data for gauss fitting.
							 ******************************************************************************/
							int newN = 0;
							for (int j = 0; j < hostCenter.length; j++) // loop over all possible centers.
							{
								if (hostCenter[j]> 0) // if the center was added.
								{									
									newN++;
								}
							}

							int loaded = 0;
							while (loaded < newN)
							{
								int maxLoad = 100000;
								if ((newN - loaded) < maxLoad)
									maxLoad = newN - loaded;



								int[] locatedCenter = new int[maxLoad]; // cleaned vector with indexes of centras.
								int[] locatedFrame = new int[maxLoad]; // cleaned vector with indexes of centras.
								int counter = 0;

								int j = loaded;
								boolean fill = true;
								while (fill)
								{
									if (hostCenter[j]> 0 )
									{
										locatedCenter[counter] = hostCenter[j];													
										locatedFrame[counter] = hostCenter[j]/(columns*rows);							
										counter++;
									}
									j ++;
									if (counter == maxLoad || j == hostCenter.length)
										fill = false;										
								}						

								double[] P = new double[counter*7];
								double[] stepSize = new double[counter*7];							
								int[] gaussVector = new int[counter*gWindow*gWindow];

								for (int i = 0; i < counter; i++)
								{
									P[i*7] = data[locatedCenter[i]];
									P[i*7+1] = 2;
									P[i*7+2] = 2;
									P[i*7+3] = 1.5;
									P[i*7+4] = 1.5;
									P[i*7+6] = 0;
									P[i*7+6] = 0;
									stepSize[i * 7] = 0.1;// amplitude
									stepSize[i * 7 + 1] = 0.25*100/inputPixelSize; // x center.
									stepSize[i * 7 + 2] = 0.25*100/inputPixelSize; // y center.
									stepSize[i * 7 + 3] = 0.25*100/inputPixelSize; // sigma x.
									stepSize[i * 7 + 4] = 0.25*100/inputPixelSize; // sigma y.
									stepSize[i * 7 + 5] = 0.19625; // Theta.
									stepSize[i * 7 + 6] = 0.01; // offset.   
									int k = locatedCenter[i] - (gWindow / 2) * (columns + 1); // upper left corner.
									j = 0;
									int loopC = 0;
									while (k <= locatedCenter[i] + (gWindow / 2) * (columns + 1)) // loop over all relevant pixels. use this loop to extract data based on single indexing defined centers.
									{
										gaussVector[i * gWindow * gWindow + j] = data[k]; // add data.
										k++;
										loopC++;
										j++;
										if (loopC == gWindow)
										{
											k += (columns - gWindow);
											loopC = 0;
										}
									} // data pulled.
								}

								CUdeviceptr deviceGaussVector 	= 		CUDA.copyToDevice(gaussVector);					
								CUdeviceptr deviceP 			= 		CUDA.copyToDevice(P);
								CUdeviceptr deviceStepSize 		= 		CUDA.copyToDevice(stepSize);							

								/******************************************************************************
								 * Gauss fitting.
								 ******************************************************************************/

								gridSizeX = (int) Math.ceil((Math.sqrt(counter)));
								gridSizeY 	= gridSizeX;
								Pointer kernelParametersGaussFit 		= Pointer.to(   
										Pointer.to(deviceGaussVector),
										Pointer.to(new int[]{counter * gWindow * gWindow}),
										Pointer.to(deviceP),																											
										Pointer.to(new double[]{counter*7}),
										Pointer.to(new short[]{(short) gWindow}),
										Pointer.to(deviceBounds),
										Pointer.to(new double[]{bounds.length}),
										Pointer.to(deviceStepSize),																											
										Pointer.to(new double[]{counter*7}),
										Pointer.to(new double[]{convCriteria}),
										Pointer.to(new int[]{maxIterations}));	

								cuLaunchKernel(fittingFcn,
										gridSizeX,  gridSizeY, 1, 	// Grid dimension
										blockSizeX, blockSizeY, 1,  // Block dimension
										0, null,               		// Shared memory size and stream
										kernelParametersGaussFit, null 		// Kernel- and extra parameters
										);
								cuCtxSynchronize(); 

								double hostOutput[] = new double[counter*7];

								// Pull data from device.
								cuMemcpyDtoH(Pointer.to(hostOutput), deviceP,
										counter*7 * Sizeof.DOUBLE);
								// Free up memory allocation on device, housekeeping.
								cuMemFree(deviceGaussVector);   
								cuMemFree(deviceP);    
								cuMemFree(deviceStepSize);							
								for (int n = 0; n < counter; n++) //loop over all particles
								{	    	
									Particle Localized = new Particle();
									Localized.include 		= 1;
									Localized.channel 		= Ch;
									Localized.frame   		= startFrame + locatedFrame[n];//locatedCenter[n]/(columns*rows);
									Localized.r_square 		= hostOutput[n*7+6];
									Localized.x				= inputPixelSize*(0.5 + hostOutput[n*7+1] + (locatedCenter[n]%columns) - Math.round((gWindow)/2));
									Localized.y				= inputPixelSize*(0.5 + hostOutput[n*7+2] + ((locatedCenter[n]/columns)%rows) - Math.round((gWindow)/2));
									Localized.z				= inputPixelSize*0;	// no 3D information.
									Localized.sigma_x		= inputPixelSize*hostOutput[n*7+3];
									Localized.sigma_y		= inputPixelSize*hostOutput[n*7+4];
									Localized.photons		= (int) (hostOutput[n*7]/totalGain[Ch-1]);
									Localized.precision_x 	= Localized.sigma_x/Math.sqrt(Localized.photons);
									Localized.precision_y 	= Localized.sigma_y/Math.sqrt(Localized.photons);
									Results.add(Localized);
								}	
								loaded += maxLoad;
							} 
						}
					} // frame loop.
					cuMemFree(deviceBounds);
				} // loop over all channels.

				ArrayList<Particle> cleanResults = new ArrayList<Particle>();
				for (int i = 0; i < Results.size(); i++)
				{
					if (Results.get(i).x > 0 &&
							Results.get(i).y > 0 &&						
							Results.get(i).sigma_x > 0 &&
							Results.get(i).sigma_y > 0 &&
							Results.get(i).precision_x > 0 &&
							Results.get(i).precision_y > 0 &&
							Results.get(i).photons > 0 && 
							Results.get(i).r_square > 0)
						cleanResults.add(Results.get(i));

				}

				/*
				 * 
				 * Remove duplicates that can occur if two center pixels has the exact same value. Select the best fit if this is the case.
				 */
				int currFrame = -1;
				int i  = 0;
				int j = 0;
				int Ch = 1;

				int pixelDistance = 2*inputPixelSize*inputPixelSize;
				while( i < cleanResults.size())
				{				
					if (cleanResults.get(i).channel > Ch)
						pixelDistance = 2*inputPixelSize*inputPixelSize;
					if( cleanResults.get(i).frame > currFrame)
					{
						currFrame = cleanResults.get(i).frame;					
						j = i+1;

					}
					while (j < cleanResults.size() && cleanResults.get(j).frame == currFrame)
					{
						if (((cleanResults.get(i).x - cleanResults.get(j).x)*(cleanResults.get(i).x - cleanResults.get(j).x) + 
								(cleanResults.get(i).y - cleanResults.get(j).y)*(cleanResults.get(i).y - cleanResults.get(j).y)) < pixelDistance)
						{					
							if (cleanResults.get(i).r_square > cleanResults.get(j).r_square)
							{
								cleanResults.get(j).include = 0;

							}else
							{
								cleanResults.get(i).include = 0;					
							}
						}
						j++;
					}

					i++;
					j = i+1;
				}
				for (i = cleanResults.size()-1; i >= 0; i--)
				{
					if (cleanResults.get(i).include == 0)
						cleanResults.remove(i);
				}

				if (modality.equals("2D"))
				{
					cleanResults = BasicFittingCorrections.compensate(cleanResults); // change 2D data to 3D data based on calibration data.
				}
				else if (modality.equals("PRILM"))
				{
					cleanResults = PRILMfitting.fit(cleanResults); // change 2D data to 3D data based on calibration data.
				}
				else if (modality.equals("Biplane"))
				{
					cleanResults = BiplaneFitting.fit(cleanResults,inputPixelSize,totalGain); // change 2D data to 3D data based on calibration data.
					columns /= 2;
				}
				else if (modality.equals("Double Helix"))
				{
					cleanResults = DoubleHelixFitting.fit(cleanResults); // change 2D data to 3D data based on calibration data.
				}
				else if (modality.equals("Astigmatism"))
				{
					cleanResults = AstigmatismFitting.fit(cleanResults); // change 2D data to 3D data based on calibration data.
				}

				ij.measure.ResultsTable tab = Analyzer.getResultsTable();
				tab.reset();		
				tab.incrementCounter();
				tab.addValue("width", columns*inputPixelSize);
				tab.addValue("height", rows*inputPixelSize);
				tab.show("Results");
				TableIO.Store(cleanResults);
				return cleanResults;
			} // end GPU computing.
		return Results;					
	}
}
