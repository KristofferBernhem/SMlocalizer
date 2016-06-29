import ij.ImagePlus;
import ij.ImageStack;
import ij.WindowManager;
import ij.plugin.PlugIn;

public class backgroundFilter_ implements PlugIn {
	public void run(String arg0){
		int W 						= 50; 	// Window width for median filtering, user input.
		ImagePlus image 			= WindowManager.getCurrentImage();  	// Aquire the selected image.		
		ImageStack stack 			= image.getStack(); 					// This should be a stack
		ImageStack CorrectedStack 	= BackgroundCorrection.medianFiltering(stack, W); 	// Median filtered background with noise level.		
		String Imtitle 				= "BackgroundCorrected_" + image.getTitle(); 		// Get the results stack new name.		
		ImagePlus FilteredImage 	=  new ImagePlus(Imtitle,CorrectedStack);					// Create new image from the result stack.
		FilteredImage.show(); 	
	}
}
