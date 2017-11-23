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


import javax.swing.JOptionPane;

import ij.IJ;
import ij.WindowManager;
import ij.plugin.PlugIn;

public class SML_Background_Correction implements PlugIn {

	public static void main(final String... args) throws Exception {
		// create the ImageJ application context with all available services				
		final ij.ImageJ ij = new ij.ImageJ();
		ij.setVisible(true);

		
		Class<?> clazz = SML_Background_Correction.class;
		String url = clazz.getResource("/" + clazz.getName().replace('.', '/') + ".class").toString();
		String pluginsDir = url.substring("file:".length(), url.length() - clazz.getName().length() - ".class".length());
		System.setProperty("plugins.dir", pluginsDir);		
		IJ.runPlugIn(clazz.getName(), "");
		}


		
	
	
	
	public void run(String arg) { // macro input arg should be comma separated with window length and computation mode.		
/*	Macro:	
 * arg = "2";  // GPU, 0 for CPU
 * run("SML Background Correction",arg);
	*/	
		String args =  ij.Macro.getOptions(); // get macro input.
		int computeModel = 0;
		if (args == null)
		{
			//Custom button text
			Object[] options = {"CPU",
								"GPU",
								"Cancel"};
			int n = JOptionPane.showOptionDialog(null,
					"Select processing method",
					"Processing",
					JOptionPane.YES_NO_CANCEL_OPTION,
					JOptionPane.PLAIN_MESSAGE,
					null,
					options,			    
					options[0]);			

			switch (n)
			{
			case 0: 	computeModel = 0; break;
			case 1: 	computeModel = 2; break;
			case 2: 	computeModel = 5; break;
			}
		}else
		{
			args = args.replaceAll("\\s+","");		// remove spaces, line separators
			String[] inputParameters = args.split(",");	// split based on ","
			computeModel = Integer.parseInt(inputParameters[0]);
		}
		if (computeModel != 5)
		{
			GetParameters parameters = new GetParameters();
			parameters.get(); // load parameters.	
			BackgroundCorrection.medianFiltering(parameters.windowWidth,WindowManager.getCurrentImage(),computeModel); // correct background.
		}
	}
}

