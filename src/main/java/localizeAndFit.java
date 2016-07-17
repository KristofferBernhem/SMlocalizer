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
	public static void run(double MinLevel, double sqDistance, int gWindow, int inputPixelSize, int minPosPixels){				
		ImagePlus LocalizeImage 			= WindowManager.getCurrentImage();  // Acquire the selected image.		
		int nChannels 						= LocalizeImage.getNChannels(); 	// Number of channels.
		int nFrames 						= LocalizeImage.getNFrames();		// Number of frames.
		ArrayList<Particle> Results 		= new ArrayList<Particle>();		// Fitted results array list.
		ArrayList<fitParameters> fitThese 	= new ArrayList<fitParameters>(); 	// arraylist to hold all fitting parameters.
		
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
//			long start = System.nanoTime();										// Timer.
			List<Future<Particle>> parallelCompute = exec.invokeAll(tasks);				// Execute computation.    
			for (int i = 0; i < parallelCompute.size(); i++){							// Loop over and transfer results.
				try {
					Results.add(parallelCompute.get(i).get());							// Add computed results to Results arraylist.
				} catch (ExecutionException e) {
					e.printStackTrace();
				}
			}
					
	//		long elapsed = System.nanoTime() - start;
			/*
			 * Do the below only for characterization, comment out once satisfied.
			 */
		/*	long startnorm = System.nanoTime();		
			for (fitParameters fr : fitThese) {
				ParticleFitter.Fitter(fr);			// Fit all found centers to gaussian.
			}
			long stopnorm = System.nanoTime();
			int sum = (int) ((stopnorm-startnorm)/1000000);

			elapsed /= 1000000;
			System.out.println(String.format("Elapsed time: %d ms", elapsed));
			System.out.println(String.format("... but compute tasks waited for total of %d ms; speed-up of %.2fx", sum, sum / (elapsed * 1d)));
			/*
			 * End characterization code.
			 */
			//results.get(1).get().channel
		} catch (InterruptedException e) {

			e.printStackTrace();
		}
		finally {
			exec.shutdown();
		}
		
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
