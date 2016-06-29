import java.util.ArrayList;

import ij.ImagePlus;
import ij.ImageStack;
import ij.WindowManager;
import ij.plugin.PlugIn;
import ij.process.ImageProcessor;

public class localize_particles_ implements PlugIn{
	
	public void run(String arg0){
		double SN 					= 1;							// Future user input, signal to noise.
		int Distance 				= 7; 							// Future user input, minimum distance between center pixels in events to be included.		
		int gWindow 				= 7;							// Window width for gaussian fit.
		double Noise 				= 1000;
		int inputPixelSize 			= 100;
		ImagePlus LocalizeImage = WindowManager.getCurrentImage();  // Acquire the selected image.
		ImageStack LocalizeStack = LocalizeImage.getStack(); 		// This should be a stack.
		ArrayList<Particle> Results = new ArrayList<Particle>();
		for (int Frame = 0; Frame < LocalizeStack.getSize();Frame++){
			ImageProcessor ImProc = LocalizeStack.getProcessor(Frame+1); 			
			Results.addAll(LocalizeEvents(ImProc,SN,Noise,Distance,gWindow,Frame+1,inputPixelSize)); // Add fitted data.
		}					
		
		TableIO.Store(Results);
	}
	
	
	/*
	 * Localize all events in the frame represented by ImageProcessor IP. Starts by finding regions of interest and follows by doing gaussian fitting of the regions.
	 */
	public static ArrayList<Particle> LocalizeEvents(ImageProcessor IP, double SN, double Noise, int Distance, int Window, int Frame,int pixelSize){
		float[][] DataArray 		= IP.getFloatArray();												// Array representing the frame.
		ArrayList<int[]> Center 	= LocalMaxima.FindMaxima(DataArray, Window, SN, Noise, Distance); 	// Get possibly relevant center coordinates.	
		ArrayList<Particle> Results = ParticleFitter.Fitter(IP, Center, Window, Frame, pixelSize);		// Fit all found centers to gaussian.
		return Results;																					// Results contain all particles located.
	} // end LocalizeEvents

}
