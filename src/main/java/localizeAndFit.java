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
import ij.process.ImageProcessor;

public class localizeAndFit {

	public static ArrayList<Particle> run(int[][][][] inputArray, int[] MinLevel, double[] sqDistance, int[] gWindow, int[] inputPixelSize, int[] minPosPixels, int[] totalGain , int selectedModel){		

		int nChannels 						= inputArray[0][0][0].length; 	// Number of channels.
		int nFrames 						= inputArray[0][0].length;		// Number of frames.
		ArrayList<Particle> Results 		= new ArrayList<Particle>();		// Fitted results array list.
		
		if (selectedModel == 0) // sequential
		{
			ArrayList<Particle> cleanResults = new ArrayList<Particle>();
			for (int Ch = 1; Ch <= nChannels; Ch++)							// Loop over all channels.
			{
			for (int Frame = 1; Frame <= nFrames;Frame++)					// Loop over all frames.
				{
				float[][] DataArray = new float[inputArray.length][inputArray[0].length];
				for (int x = 0; x < inputArray.length; x++)
				{
					for (int y = 0; y < inputArray[0].length; y++)
					{
						DataArray[x][y] = inputArray[x][y][Frame-1][Ch - 1];
					} // y loop.
				} // x loop.

				ArrayList<int[]> Center 	= LocalMaxima.FindMaxima(DataArray, gWindow[Ch-1], MinLevel[Ch-1], sqDistance[Ch-1], minPosPixels[Ch-1]); 	// Get possibly relevant center coordinates.

				for (int i = 0; i < Center.size(); i++)
				{
					int[] dataFit = new int[gWindow[Ch-1]*gWindow[Ch-1]];							// Container for data to be fitted.
					int[] Coord = Center.get(i);									// X and Y coordinates for center pixels to be fitted.
					
					for (int j = 0; j < gWindow[Ch-1]*gWindow[Ch-1]; j++)
					{
						int x =  Coord[0] - Math.round((gWindow[Ch-1])/2) +  (j % gWindow[Ch-1]);
						int y =  Coord[1] - Math.round((gWindow[Ch-1])/2) +  (j / gWindow[Ch-1]);
						dataFit[j] = inputArray[x][y][Frame-1][Ch - 1];
					} // pull out data for this fit.
					fitParameters fitThese = new fitParameters(Coord, 
							dataFit,
							Ch,
							Frame,
							inputPixelSize[Ch-1],
							gWindow[Ch-1],
							totalGain[Ch-1]);
					Particle tempParticle = ParticleFitter.Fitter(fitThese);
					if (tempParticle.sigma_x > 0 &&
							tempParticle.sigma_y > 0 &&
							tempParticle.precision_x > 0 &&
							tempParticle.precision_y > 0 &&
							tempParticle.photons > 0 && 
							tempParticle.r_square > 0)
					cleanResults.add(tempParticle);
					} // loop over all located centers from this frame.									
				} // loop over all frames.					
			} // loop over all channels.
			return cleanResults; // end parallel computation by returning results.
		}else // sequential 
			if (selectedModel == 1) // parallel
		{
			ArrayList<fitParameters> fitThese 	= new ArrayList<fitParameters>(); 	// arraylist to hold all fitting parameters.
			for (int Ch = 1; Ch <= nChannels; Ch++)							// Loop over all channels.
			{
			for (int Frame = 1; Frame <= nFrames;Frame++)					// Loop over all frames.
				{
				float[][] DataArray = new float[inputArray.length][inputArray[0].length];
				for (int x = 0; x < inputArray.length; x++)
				{
					for (int y = 0; y < inputArray[0].length; y++)
					{
						DataArray[x][y] = inputArray[x][y][Frame-1][Ch - 1];
					} // y loop.
				} // x loop.

				ArrayList<int[]> Center 	= LocalMaxima.FindMaxima(DataArray, gWindow[Ch-1], MinLevel[Ch-1], sqDistance[Ch-1], minPosPixels[Ch-1]); 	// Get possibly relevant center coordinates.

				for (int i = 0; i < Center.size(); i++)
				{
					int[] dataFit = new int[gWindow[Ch-1]*gWindow[Ch-1]];							// Container for data to be fitted.
					int[] Coord = Center.get(i);									// X and Y coordinates for center pixels to be fitted.
					
					for (int j = 0; j < gWindow[Ch-1]*gWindow[Ch-1]; j++)
					{
						int x =  Coord[0] - Math.round((gWindow[Ch-1])/2) +  (j % gWindow[Ch-1]);
						int y =  Coord[1] - Math.round((gWindow[Ch-1])/2) +  (j / gWindow[Ch-1]);
						dataFit[j] = inputArray[x][y][Frame-1][Ch - 1];
					} // pull out data for this fit.
					fitThese.add(new fitParameters(Coord, 
							dataFit,
							Ch,
							Frame,
							inputPixelSize[Ch-1],
							gWindow[Ch-1],
							totalGain[Ch-1]));
				} // loop over all located centers from this frame.									
			} // loop over all frames.					
		} // loop over all channels.
		
		List<Callable<Particle>> tasks = new ArrayList<Callable<Particle>>();	// Preallocate.
		for (final fitParameters object : fitThese) {							// Loop over and setup computation.
			Callable<Particle> c = new Callable<Particle>() {					// Computation to be done.
				@Override
				public Particle call() throws Exception {
					return ParticleFitter.Fitter(object);						// Actual call for each parallel process.
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
		for (int i = 0; i < Results.size(); i++)
		{
			if (Results.get(i).sigma_x > 0 &&
					Results.get(i).sigma_y > 0 &&
					Results.get(i).precision_x > 0 &&
					Results.get(i).precision_y > 0 &&
					Results.get(i).photons > 0 && 
					Results.get(i).r_square > 0)
			cleanResults.add(Results.get(i));
				
		}
		return cleanResults; // end parallel computation by returning results.
	}else // end parallel. 
		if (selectedModel == 2) // GPU 
	{
			return Results;
	} // end GPU computing.

		return Results;
		
			
	}

	/*
	 * Generate fitParameter objects by finding local maximas seperated by sqDistance of atleast MinLevel center pixel intensity. 
	 * Returns fitParameters for subsequent gaussian fitting.
	 */
	public static ArrayList<fitParameters> LocalizeEvents(ImageProcessor IP, int MinLevel, double sqDistance, int Window, int Frame, int Channel, int pixelSize, int minPosPixels, int totalGain){
		float[][] DataArray 		= IP.getFloatArray();												// Array representing the frame.
		ArrayList<int[]> Center 	= LocalMaxima.FindMaxima(DataArray, Window, MinLevel, sqDistance,minPosPixels); 	// Get possibly relevant center coordinates.
		ArrayList<fitParameters> fitThese = new ArrayList<fitParameters>();
		for (int i = 0; i < Center.size(); i++){
			int[] dataFit = new int[Window*Window];							// Container for data to be fitted.
			int[] Coord = Center.get(i);									// X and Y coordinates for center pixels to be fitted.
			
			for (int j = 0; j < Window*Window; j++)
			{
				int x =  Coord[0] - Math.round((Window)/2) +  (j % Window);
				int y =  Coord[1] - Math.round((Window)/2) +  (j / Window);
				dataFit[j] = (int) IP.getf(x,y);
			}
			fitThese.add(new fitParameters(Coord, 
					dataFit,
					Channel,
					Frame,
					pixelSize,
					Window,
					totalGain));
		}
		return fitThese;																					// Results contain all particles located.
	} // end LocalizeEvents
}
