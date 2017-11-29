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
import ij.plugin.PlugIn;

public class SML_About_ implements PlugIn
{

	public static void main(final String... args) throws Exception {
		// create the ImageJ application context with all available services				
		final ij.ImageJ ij = new ij.ImageJ();
		ij.setVisible(true);

		
		Class<?> clazz = SML_About_.class;
		String url = clazz.getResource("/" + clazz.getName().replace('.', '/') + ".class").toString();
		String pluginsDir = url.substring("file:".length(), url.length() - clazz.getName().length() - ".class".length());
		System.setProperty("plugins.dir", pluginsDir);		
		IJ.runPlugIn(clazz.getName(), "");
		}

	public void run(String arg) {
		IJ.showMessage("SMLocalizer - v2.1.0",
				"SMLocalizer plugin developed by Kristoffer Bernhem, KTH Royal Institute of Technology, Sweden.\n \n"+
				"Macro run of the plugin can be done for batch analysis of data. Macros can be found in the Utilities folder and accept the following arguments:\n" +
				"run('SML Background Correction',args); args = '0 or 2' where 0 tells the plugin to run on CPU and 2 to run on GPU. \n"+
				"run('SML Drift Correction',args); args = '0 or 2' where 0 tells the plugin to run on CPU and 2 to run on GPU. \n"+
				"run('SML Localize Particles',args); args = '0 or 2' where 0 tells the plugin to run on CPU and 2 to run on GPU. \n"+
				"run('SML Drift Correction',args); args = '0 or 2' where 0 tells the plugin to run on CPU and 2 to run on GPU. \n"+
				"run('SML Align Channels',args); args = '0 or 2' where 0 tells the plugin to run on CPU and 2 to run on GPU. \n"+
				"run('SML Render Image',args); args = 'pixel_xy, pixel_z' pixel_xy is the xy voxel size and pixel_z is the z voxel size. \n"+
				"run('SML Process',args); args = '0 or 2, Mode, pixel_xy, pixel_z' where 0 tells the plugin to run on CPU and 2 to run on GPU, "
				+ "Mode is the modality used, either 2D, Biplane, Double Helix, Astigmatism or PRILM. \n\n"+
				"For all parameters the macro commands will run on the most recently stored ones set in the main GUI of SMLocalizer. \n "+
				"A specific parameter can be set for a given channel using the commands found in 'Parameter list' in the utility section. \n\n"+			
				"Publication available through https://doi.org/10.1093/bioinformatics/btx553 \n \n"+
				"We kindly request that you cite the publication whenever presenting or publishing results based on SMLocalizer\n \n");		
	}		
}
