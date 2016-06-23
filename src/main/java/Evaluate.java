import java.io.File;
import java.util.ArrayList;

import ij.IJ;

//import org.scijava.plugin.Plugin;

import ij.ImagePlus;
import ij.ImageStack;
import ij.WindowManager;
import ij.plugin.PlugIn;
import ij.process.FloatProcessor;
import ij.process.ImageProcessor;
import net.imagej.Dataset;
import net.imagej.ImageJ;

/** Loads and displays a dataset using the ImageJ API. */
//@Plugin(menuPath = "Plugins>Examples>Dataset", type = null)


public class Evaluate implements PlugIn {

	public static void main(final String... args) throws Exception {
		// create the ImageJ application context with all available services
		final ImageJ ij = new ImageJ();
		
		// ask the user for a file to open
		final File file = ij.ui().chooseFile(null, "open");

		// load the dataset
		final Dataset dataset = ij.scifio().datasetIO().open(file.getPath());

		// display the dataset
		ij.ui().show(dataset);
		ij.ui().showUI();

		
		// Find local extreme. 
		
		//float value = ip.getf(0, 0);
		//System.out.println(width + "x" + height + "x" + nFrames );

		Class<?> clazz = Evaluate.class;
		String url = clazz.getResource("/" + clazz.getName().replace('.', '/') + ".class").toString();
		String pluginsDir = url.substring("file:".length(), url.length() - clazz.getName().length() - ".class".length());
		System.setProperty("plugins.dir", pluginsDir);
		IJ.runPlugIn(clazz.getName(), "");
		
	}

	@Override
	public void run(String arg0) {
		double SN = 0.05; 		// Future user input, signal to noise.
		int Distance = 5; 	// Future user input, minimum distance between center pixels in events to be included.
		int W = 50; 		// Window width for median filtering, user input.
		
		ImagePlus image = WindowManager.getCurrentImage();  // Aquire the selected image.
		
		ImageStack stack = image.getStack(); 				// This should be a stack
		int nFrames = image.getNFrames();					// Number of timepoints.
		int width = image.getHeight();						// Frame height (x);
		int height = image.getWidth(); 						// Frame width (y);
		//int StackS = image.getImageStackSize(); // Check multicolor stack to see that loading is ok.
		long start = System.nanoTime();
					
		ImageStack ImStack = stack.convertToFloat(); 		// Change type, we want to work in double format.		
		double[][][] dataArray = new double[width][height][nFrames]; // Array to put all data in.
		for (int Frame = 1; Frame <= nFrames; Frame++){ 		// Loop over the image and recast pixel data to double.
			float[] ImFrame = (float[]) ImStack.getPixels(Frame);
			int col = 0;
			int row = 0;
			for (int i = 0;i < ImFrame.length; i ++){
				dataArray[col][row][Frame-1] = ImFrame[i];
				col++;
				if (col == width){
					col = 0;
					row++;
				}			
			}		
		} 

		double[] Corrected = BackgroundCorrection.medianFiltering2(dataArray, W); // Median filtered background with noise level.
		
		ImageStack newStack = new ImageStack(width,height);				// Create new imagestack to hold background corrected data.
		for (int Frames = 0; Frames < nFrames; Frames++){ 				// Loop over all frames.
			float[] slice = new float[width*height]; 					// Hold pixel intensities for this slice.
			int count = 0; 												// Use to add data to the slice.
			for (int i = width*height*Frames; i < width*height*(Frames+1); i++){ // Loop over all pixels in the slice.
				slice[count] = (float) Corrected[i]; 					// Add pixel data.
				if (slice[count] < 0){ 									// Pixel data might be negative, set to 0 if so.
					slice[count] = 0;
				}	
				count++;												// Step forward in the slice.
			}
			newStack.addSlice(new FloatProcessor(width,height,slice)); 	// After slice has been populated, add it to the results stack at the end.
		}
		String Imtitle = "BackgroundCorrected_" + image.getTitle(); // Get the results stack new name.		
		ImagePlus FilteredImage = new ImagePlus(Imtitle,newStack);	// Create new image from the result stack.	
		FilteredImage.show(); 										// Make visible.

		/*
		 * The first part of the plugin is now done, data is background filtered. User can save the mid point result.
		 * Other filter algorithms can be applied to the corrected image stack if required. Once done, the next step in the plugin will run.
		 * 
		 */
		
		/*
		 * Next step in plugin, localize each event based on user input threshold settings. 
		 * 
		 */
		
		
		
		
		ImagePlus LocalizeImage = WindowManager.getCurrentImage();  // Aquire the selected image.

		ImageStack LocalizeStack = LocalizeImage.getStack(); 		// This should be a stack.
		for (int Frame = 0; Frame < LocalizeStack.getSize();Frame++){
			ImageProcessor ImProc = LocalizeStack.getProcessor(Frame+1); 
			double Noise = 0;
			ArrayList<int[]> Results = LocalizeEvents(ImProc,SN,Noise,Distance);
		}
		
		
		/*
		Count = 0;
		//int Count = 0;
		double[] NoiseVector = new double[nFrames];
		for (int i = width*height*nFrames; i < width*height*nFrames + nFrames; i++){			
			NoiseVector[Count] = Corrected[i];
			Count++;
		}
/*		if (nFrames > 2*W+1){ // Check that the user loaded a correct image stack. 		
			dataArray = BackgroundCorrection.medianFiltering(dataArray, W); // Median filtered background.
		}  
		double Noise = 1000; /*
		int Found = 0;
		for (int Frame = 0; Frame < nFrames; Frame ++){
			double[][] dataSlice = new double[width][height];
			for (int col = 0; col < width; col++){
				for (int row = 0; row < height; row++){
					 dataSlice[col][row] = dataArray[col][row][Frame];
				}
			}
			ArrayList<int[]> Result = LocalMaxima.FindMaxima(dataSlice, SN, NoiseVector[Frame], Distance);
			
			// Add localization here.
			
			Found += Result.size();
		}
		*/
		long stop = System.nanoTime();
		//System.out.println(Found + " events in " + (stop-start)/1000000 + "ms");
		
		//System.exit(0);
	}

	public static ArrayList<int[]> LocalizeEvents(ImageProcessor IP, double SN, double Noise, int Distance){
		float[][] DataArray = IP.getFloatArray();
		
		ArrayList<int[]> Results = LocalMaxima.FindMaxima(DataArray, SN, Noise, Distance); // Get possibly relevant center coordinates.
		/*
		 * Go through Results and do gaussian fitting to each. Create object Event:
		 * .x
		 * .y
		 * .sx
		 * .sy
		 * .photons
		 * .chi^2
		 * .precision x
		 * .precision y
		 */
		
		Particle Event = new Particle();
		//ArrayList<Particle> Res = new ArrayList<Particle>();
		
		return Results;
					
	}
}
