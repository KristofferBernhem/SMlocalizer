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
import net.imagej.ImageJ;
import ij.Prefs;


public class SMLocalizer_ implements PlugIn {

	public static void main(final String... args) throws Exception {
		// create the ImageJ application context with all available services

		final ImageJ ij = new ImageJ();
		ij.ui().showUI();
		createTestDataSet.stack(64, 64, 1000, 1, 1); // get dataset.
		double[] drift = {0.1,0.1};
	//	createTestDataSet.ParticleList(6400, 6400, 10000, 500, drift);
		Class<?> clazz = SMLocalizer_.class;
		String url = clazz.getResource("/" + clazz.getName().replace('.', '/') + ".class").toString();
		String pluginsDir = url.substring("file:".length(), url.length() - clazz.getName().length() - ".class".length());
		System.setProperty("plugins.dir", pluginsDir);				
		IJ.runPlugIn(clazz.getName(), "");
		
		/*
		 * how to store variables:
		 */
	/*	Prefs.set(("SMlocalizer.newTestValue." + 1), 21);
		Prefs.set(("SMlocalizer.newTestValue." + 3), 93);		
		int MyNr = (int) Prefs.get("SMlocalizer.newTestValue.1", 0);
		int MyNr2 = (int) Prefs.get("SMlocalizer.newTestValue.3", 0);
		System.out.println(MyNr + " x " + MyNr2);
		*/
	}


	@Override
	public void run(String arg0) {
		String args[] = null;
		SMLocalizerGUI.main(args); //call GUI.
		
		
		
	}

}