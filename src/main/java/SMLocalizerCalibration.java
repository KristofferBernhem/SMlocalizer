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

import ij.IJ;
import ij.plugin.PlugIn;

public class SMLocalizerCalibration implements PlugIn {

	public static void main(final String... args) throws Exception {
		// create the ImageJ application context with all available services				

		/*
		 * notes for 3D PRILM/Double Helix fitting.
		 * 
		 * Workflow:
		 * Background correct and fit each lobe independently
		 * In each frame, find closest neighbor. If no neighbor within OK range (from calibration file) discard point.
		 * For each pair, calculate angle between points. Check this against lookup table from calibration file for z. 
		 * TODO How to get eqvivalent sigma for z. 
		 * TODO How to get eqvivalent precision for z.
		 * 
		 * Calculate x,y at mean value of the two located lobes. 
		 * 
		 * Calibration file is generated by separate plugin that only generates a PRILM lookup table (in settings file). 
		 * Bead sample with sweeping focus in small, user provided, steps is required as input. 
		 * Calculate angle for each step and generate lookuptable with angle vs z-depth for the relevant fits (good original fit data).
		 * 
		 *  include dropdown menu for user to select which imaging mode the images are from.
		 *  
		 *  Add string from the modality choice to the localize calls and add call to modality specific function for 2D->3D translation after initial fitting, calling with the particle list. Return new particle list and update results table.

			short dx = 50; // diff in x dimension.
			short dy = 46; // diff in y dimension.
			double zAngle = (Math.atan2(dy, dx)); // angle between points and horizontal axis.
		 */

		final ij.ImageJ ij = new ij.ImageJ();
		ij.setVisible(true);
		Class<?> clazz = SMLocalizerCalibration.class;
		String url = clazz.getResource("/" + clazz.getName().replace('.', '/') + ".class").toString();
		String pluginsDir = url.substring("file:".length(), url.length() - clazz.getName().length() - ".class".length());
		System.setProperty("plugins.dir", pluginsDir);				
		IJ.runPlugIn(clazz.getName(), "");
	}
	@Override
	public void run(String arg0) {
		String args[] = null;
		CalibrationGUI.main(args); //call GUI.

	}

}