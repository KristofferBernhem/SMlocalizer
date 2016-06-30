

import ij.ImagePlus;
import ij.ImageStack;
import ij.WindowManager;


/*
 * ToDo: Add functionality for multi channel stacks, looping over each channel separately, possibly calling medianFiltering multipiple times.
 */

public class filterBackground {
	public static void run(int W){
		ImagePlus image 			= WindowManager.getCurrentImage();  				// Aquire the selected image.		
		ImageStack stack 			= image.getStack(); 								// This should be a stack
		ImageStack CorrectedStack 	= BackgroundCorrection.medianFiltering(stack, W); 	// Median filtered background with noise level.		
		String Imtitle 				= "BackgroundCorrected_" + image.getTitle(); 		// Get the results stack new name.		
		ImagePlus FilteredImage 	=  new ImagePlus(Imtitle,CorrectedStack);			// Create new image from the result stack.
		FilteredImage.show(); 															// Output.
	}
}


