/* Copyright 2017 Kristoffer Bernhem.
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
/**
 *
 * @author kristoffer.bernhem@gmail.com
 */


import ij.IJ;
import ij.WindowManager;
import ij.plugin.PlugIn;

public class Background_correction implements PlugIn {

	public static void main(final String... args) throws Exception {
		// create the ImageJ application context with all available services				
		final ij.ImageJ ij = new ij.ImageJ();
		ij.setVisible(true);
		Class<?> clazz = SMLocalizer_.class;
		String url = clazz.getResource("/" + clazz.getName().replace('.', '/') + ".class").toString();
		String pluginsDir = url.substring("file:".length(), url.length() - clazz.getName().length() - ".class".length());
		System.setProperty("plugins.dir", pluginsDir);				
		IJ.runPlugIn(clazz.getName(), "");
		}


		
	
	
	
	public void run(String arg) { // macro inptut arg should be comma separated with window length and computation mode.		
		String args =  ij.Macro.getOptions(); // get macro input.
		args = args.replaceAll("\\s+","");		// remove spaces, line separators
		String[] inputParameters = args.split(",");	// split based on ","
		if (inputParameters.length >= 2) // if we got precisely two inputs.
		{
			final int[] Window = new int[10]; // pull out window length.
			for (int i = 0; i < inputParameters.length-1; i++)
				Window[i] = Integer.parseInt(inputParameters[i]); // pull out window length.
			if (inputParameters.length-1 < 10) // if we did not populate all entries, add default to others.
			{
				for (int i = inputParameters.length-1; i < 10; i++)
					Window[i] = 50; // default window length.
			}
			int selectedModel = Integer.parseInt(inputParameters[inputParameters.length-1]);			// pull out mode selection.
			BackgroundCorrection.medianFiltering(Window,WindowManager.getCurrentImage(),selectedModel); // correct background.		
		}else if (inputParameters.length == 1)
		{
			final int[] Window = {50,50,50,50,50,50,50,50,50,50}; // pull out window length.
			int selectedModel = 0; // CPU
			BackgroundCorrection.medianFiltering(Window,WindowManager.getCurrentImage(),selectedModel); // correct background.
		}else
		{
			final int[] Window = {50,50,50,50,50,50,50,50,50,50}; 	// default.
			int selectedModel = 0; 				// CPU
			BackgroundCorrection.medianFiltering(Window,WindowManager.getCurrentImage(),selectedModel); // correct background.
		}
				
		
		
	}
}

