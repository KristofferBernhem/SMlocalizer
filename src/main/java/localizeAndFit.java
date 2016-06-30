import java.util.ArrayList;

import ij.ImagePlus;
import ij.ImageStack;
import ij.WindowManager;
import ij.process.ImageProcessor;

/*
 * ToDo: add functionallity for multi channel data.
 */

public class localizeAndFit {
	public static void run(double SN, int Distance, int gWindow, double Noise, int inputPixelSize){
		double Channels = 1; 										// Get this from the image stack.
		ImagePlus LocalizeImage = WindowManager.getCurrentImage();  // Acquire the selected image.
		ImageStack LocalizeStack = LocalizeImage.getStack(); 		// This should be a stack.
		ArrayList<Particle> Results = new ArrayList<Particle>();
		for (double Ch = 1; Ch <= Channels; Ch++){
			for (int Frame = 0; Frame < LocalizeStack.getSize();Frame++){
				ImageProcessor ImProc = LocalizeStack.getProcessor(Frame+1); 			
				Results.addAll(LocalizeEvents(ImProc,SN,Noise,Distance,gWindow,Frame+1, Ch, inputPixelSize)); // Add fitted data.
			}					
		}
		TableIO.Store(Results);
	}


	/*
	 * Localize all events in the frame represented by ImageProcessor IP. Starts by finding regions of interest and follows by doing gaussian fitting of the regions.
	 */
	public static ArrayList<Particle> LocalizeEvents(ImageProcessor IP, double SN, double Noise, int Distance, int Window, int Frame, double Channel, int pixelSize){
		float[][] DataArray 		= IP.getFloatArray();												// Array representing the frame.
		ArrayList<int[]> Center 	= LocalMaxima.FindMaxima(DataArray, Window, SN, Noise, Distance); 	// Get possibly relevant center coordinates.	
		ArrayList<Particle> Results = ParticleFitter.Fitter(IP, Center, Window, Frame, Channel, pixelSize);		// Fit all found centers to gaussian.
		return Results;																					// Results contain all particles located.
	} // end LocalizeEvents
}
