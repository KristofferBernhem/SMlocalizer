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
//Fit3D.fit(MinLevel,gWindow,inputPixelSize,totalGain,maxSigma)
public class Fit3D {
	public static ArrayList<Particle> fit(int[] MinLevel,int gWindow,int inputPixelSize, int[] totalGain, double maxSigma)
	{
		ImagePlus image 					= WindowManager.getCurrentImage();
		int columns 						= image.getWidth();
		int rows 							= image.getHeight();

		int nChannels 						= image.getNChannels(); 	// Number of channels.
		int nFrames 						= image.getNFrames();
		if (nFrames == 1)
			nFrames 						= image.getNSlices();
		
		int minPosPixels = (gWindow*gWindow - 4); // update to relevant numbers for this modality.
		ImageProcessor IP = image.getProcessor();
		ArrayList<Particle> Results 		= new ArrayList<Particle>();		// Fitted results array list.
		ArrayList<fitParameters> fitThese 	= new ArrayList<fitParameters>(); 	// arraylist to hold all fitting parameters.
		
		for (int Ch = 1; Ch <= nChannels; Ch++)							// Loop over all channels.
		{
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
				ArrayList<int[]> Center 	= LocalMaxima.FindMaxima(IP, gWindow, MinLevel[Ch-1], minPosPixels); 	// Get possibly relevant center coordinates.
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
		int pixelDistance = 2*inputPixelSize*inputPixelSize;
		while( i < cleanResults.size())
		{							
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
		ij.measure.ResultsTable tab = Analyzer.getResultsTable();
		tab.reset();		
		tab.incrementCounter();
		tab.addValue("width", columns*inputPixelSize);
		tab.addValue("height", rows*inputPixelSize);
		tab.show("Results");
		
		TableIO.Store(cleanResults);		
		return cleanResults; // end parallel computation by returning results.
	}
}
