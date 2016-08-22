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

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

import ij.ImagePlus;
import ij.ImageStack;
import ij.WindowManager;
import ij.process.ImageProcessor;


public class localizeAndFit {
	public static void run(double MinLevel, double sqDistance, int gWindow, int inputPixelSize, int minPosPixels, boolean GPU){				
		ImagePlus LocalizeImage 			= WindowManager.getCurrentImage();  // Acquire the selected image.		
		int nChannels 						= LocalizeImage.getNChannels(); 	// Number of channels.
		int nFrames 						= LocalizeImage.getNFrames();		// Number of frames.
		ArrayList<Particle> Results 		= new ArrayList<Particle>();		// Fitted results array list.
		ArrayList<fitParameters> fitThese 	= new ArrayList<fitParameters>(); 	// arraylist to hold all fitting parameters.
		double z0 = 0;
		double sigma_z = 0;
		if (nChannels > 1){ // If multichannel.
			for (int Ch = 1; Ch <= nChannels; Ch++){							// Loop over all channels.
				for (int Frame = 1; Frame <= nFrames;Frame++){					// Loop over all frames.
					LocalizeImage.setPosition(Ch, 1, Frame);					// Update position in the image.
					ImageProcessor ImProc = LocalizeImage.getProcessor();		// Get current image processor.
					fitThese.addAll(LocalizeEvents(								// Add fitted data.
							ImProc,
							MinLevel,
							sqDistance,
							gWindow,
							Frame, 
							Ch, 
							inputPixelSize, 
							minPosPixels)); 
				}					
			}
		}else { // if single channel data.
			int Ch = 1;															// Only one channel.			
			ImageStack stack = LocalizeImage.getStack(); 						// Get stack.
			nFrames = stack.getSize();											// Number of timepoints, getNFrames() yiels 1.
			for (int Frame = 1; Frame <= nFrames;Frame++){						// Loop over all frames.				
				ImageProcessor ImProc = stack.getProcessor(Frame);				// Get current image processor.
				fitThese.addAll(LocalizeEvents(									// Add fitted data.
						ImProc,
						MinLevel,
						sqDistance,
						gWindow,
						Frame, 
						Ch, 
						inputPixelSize, 
						minPosPixels)); 
			}
		}

		if (GPU){
			// TODO: generate input to ptx code and run ptx code.
			/* required input:
			 * device_GaussVector: data to be fitted, sequentially in a 1 dimensional array.
			 * device_ParameterVector: number of fits * 7 long vector containing initial guess. This is the return vector with optimized results.
			 * gWindow: fitted window width.
			 * device_bounds: boundry conditions.
			 * device_steps: stepsize, in pixel fractions.
			 */
			
            // low-high for each parameter. Bounds are inclusive.
            double[] bounds = {
                          0.8, 	1.3,         // amplitude, should be close to center pixel value. Add +/-20 % of center pixel, not critical for performance.
                          1.5,  2.5,        // x coordinate. Center has to be around the center pixel if gaussian distributed.
                          1.5,  2.5,        // y coordinate. Center has to be around the center pixel if gaussian distributed.
                          0.5,  2.5,        // sigma x. Based on window size.
                          0.5,  2.5,        // sigma y. Based on window size.
                            0, .785,        // Theta. 0.785 = pi/4. Any larger and the same result can be gained by swapping sigma x and y, symetry yields only positive theta relevant.
                         -0.5,  0.5};        // offset, best estimate, not critical for performance.
            
            // steps is the most critical for processing time. Final step is 1/25th of these values. 
            double[] steps = {
                            0.25,             // amplitude, make final step 1% of max signal.
                            0.25,           // x step, final step = 1 nm.
                            0.25,           // y step, final step = 1 nm.
                            0.5,            // sigma x step, final step = 2 nm.
                            0.5,            // sigma y step, final step = 2 nm.
                            0.19625,        // theta step, final step = 0.00785 radians. Start value == 25% of bounds.
                            0.025};            // offset, make final step 0.1% of signal.
            double [] InitialGuess = {0, 	// Amplitude.
    				(gWindow-1)/2, 			// x center.
    				(gWindow-1)/2, 			// y center.
    				(gWindow-1)/3.5, 		// sigma x.
    				(gWindow-1)/3.5, 		// sigma y.
    				0, 						// theta.
    				0 						// offset.
            		
            };
            int N = fitThese.size(); 						// number of objects to fit.
            int dataSize = gWindow*gWindow;
            int[] GaussVector = new int[N*gWindow]; 		// initiate.
            double[] ParameterVector = new double[N*7];   	// initiate.
            for (int n = 0; n < N; n++){
            	int[] data =  fitThese.get(n).data;
            	for (int i = 0; i < dataSize; i++)
            	{
            		GaussVector[n*dataSize + i] = data[i]; //populate GaussVector.            		
            	}
            	ParameterVector[n*dataSize] = data[gWindow*(gWindow-1)/2 + (gWindow-1)/2];
            	for (int i = 1; i < InitialGuess.length; i++)
            	{
            		ParameterVector[n*7 + i] = InitialGuess[i];
            	}
            }
            
            // gpu input is now set up. Transfer data to gpu and run ptx code.
           
            
            // TODO: add gpu transfer and ptx call.
            // get parameter vector back from gpu.
            // TODO: add gpu retrieval of ParameterVector.
            // generate output.
            for (int n = 0; n < N; n++) // loop over all particles sent to gpu for fitting.
            {            	
            	Particle Localized = new Particle(); // create new particle.
        		Localized.include 		= 1;
        		Localized.channel 		= fitThese.get(n).channel;
        		Localized.frame   		= fitThese.get(n).frame;
        		Localized.r_square 		= ParameterVector[n*dataSize + 6];
        		Localized.x				= inputPixelSize*(ParameterVector[n*dataSize + 1] + fitThese.get(n).Center[0] - Math.round((gWindow-1)/2));
        		Localized.y				= inputPixelSize*(ParameterVector[n*dataSize + 2] + fitThese.get(n).Center[1] - Math.round((gWindow-1)/2));
        		Localized.z				= inputPixelSize*z0;
        		Localized.sigma_x		= inputPixelSize*ParameterVector[n*dataSize + 3];
        		Localized.sigma_y		= inputPixelSize*ParameterVector[n*dataSize + 4];
        		Localized.sigma_z		= inputPixelSize*sigma_z;
        		Localized.photons		= ParameterVector[n*dataSize];
        		Localized.precision_x 	= Localized.sigma_x/Localized.photons;
        		Localized.precision_y 	= Localized.sigma_y/Localized.photons;
        		Localized.precision_z 	= Localized.sigma_z/Localized.photons; 
        		Results.add(Localized); // add current.
            }
            
		}else{
			
			List<Callable<Particle>> tasks = new ArrayList<Callable<Particle>>();	// Preallocate.
			for (final fitParameters object : fitThese) {							// Loop over and setup computation.
				Callable<Particle> c = new Callable<Particle>() {					// Computation to be done.
					@Override
					public Particle call() throws Exception {
						return ParticleFitter.Fitter(object);						// Actual call for each parallel process.
					}
				};
				tasks.add(c);														// Que this task.
			} 

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
			}
		} // end CPU bound computing.
		TableIO.Store(Results);												// Return and display results to user.
	}


	/*
	 * Generate fitParameter objects by finding local maximas seperated by sqDistance of atleast MinLevel center pixel intensity. 
	 * Returns fitParameters for subsequent gaussian fitting.
	 */
	public static ArrayList<fitParameters> LocalizeEvents(ImageProcessor IP, double MinLevel, double sqDistance, int Window, int Frame, int Channel, int pixelSize, int minPosPixels){
		float[][] DataArray 		= IP.getFloatArray();												// Array representing the frame.
		ArrayList<int[]> Center 	= LocalMaxima.FindMaxima(DataArray, Window, MinLevel, sqDistance,minPosPixels); 	// Get possibly relevant center coordinates.
		ArrayList<fitParameters> fitThese = new ArrayList<fitParameters>();
		for (int i = 0; i < Center.size(); i++){
			int[] dataFit = new int[Window*Window];							// Container for data to be fitted.
			int[] Coord = Center.get(i);									// X and Y coordinates for center pixels to be fitted.
			int count = 0;	
			for (int x = Coord[0]-(Window-1)/2; x<= Coord[0] + (Window-1)/2; x++){ 	// Get all pixels for the region.
				for (int y = Coord[1]-(Window-1)/2; y<= Coord[1] + (Window-1)/2; y++){
					dataFit[count] = (int) IP.getf(x, y);	
					count++;
				}
			}					
			fitThese.add(new fitParameters(Coord, 
					dataFit,
					Channel,
					Frame,
					pixelSize,
					Window));
		}

		return fitThese;																					// Results contain all particles located.
	} // end LocalizeEvents
}
