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


