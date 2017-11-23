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
import ij.plugin.PlugIn;

public class SML_Render_Image implements PlugIn {

	public static void main(final String... args) throws Exception {
		// create the ImageJ application context with all available services				
		final ij.ImageJ ij = new ij.ImageJ();
		ij.setVisible(true);
		Class<?> clazz = SML_Render_Image.class;
		String url = clazz.getResource("/" + clazz.getName().replace('.', '/') + ".class").toString();
		String pluginsDir = url.substring("file:".length(), url.length() - clazz.getName().length() - ".class".length());
		System.setProperty("plugins.dir", pluginsDir);				
		IJ.runPlugIn(clazz.getName(), "");
	}






	public void run(String arg) { // macro inptut arg should be comma separated with window length and computation mode.		
		/*	Macro:	
		 * arg = "5,10";  pixel size xy,z
		 * run("SML Render Image",arg);
		 */	
		
		String args =  ij.Macro.getOptions(); // get macro input.
		int[] outputPixelSize = new int[2];	
		if (args == null)
		{
			//Custom button text

			String s = (String)JOptionPane.showInputDialog(
					null,
					"Set voxelsize as xy,z",
					"Voxel size [nm]",
					JOptionPane.PLAIN_MESSAGE,
					null,
					null,
					null);			
			if (s != null)
			{
				s = s.replaceAll("\\s+","");		// remove spaces, line separators
				String[] inputParameters = s.split(",");	// split based on ","
				outputPixelSize[0] 		= Integer.parseInt(inputParameters[0]);		
				if (inputParameters.length >1)
					outputPixelSize[1] 		= Integer.parseInt(inputParameters[1]);		
			}
		}else
		{
			args = args.replaceAll("\\s+","");		// remove spaces, line separators
			String[] inputParameters = args.split(",");	// split based on ","
			outputPixelSize[0] 		= Integer.parseInt(inputParameters[0]);		
			outputPixelSize[1] 		= Integer.parseInt(inputParameters[1]);		
		}
		if (outputPixelSize[0] != 0)
		{

			try // in case no results table are available.
			{
				GetParameters parameters = new GetParameters();
				parameters.get(); // load parameters.				
				cleanParticleList.run(parameters.lb,parameters.ub,parameters.includeParameters); // clean out results table.			
				RenderIm.run(parameters.doRender,outputPixelSize,parameters.doGaussianSmoothing); // Not 3D yet, how to implement? Need to find out how multi channel images are organized for multi channel functions.
			}
			finally
			{

			}

		}	
	}
}

