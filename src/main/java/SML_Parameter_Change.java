import ij.IJ;
import ij.plugin.PlugIn;

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
public class SML_Parameter_Change implements PlugIn {

	public static void main(final String... args) throws Exception {
		// create the ImageJ application context with all available services				
		final ij.ImageJ ij = new ij.ImageJ();
		ij.setVisible(true);


		Class<?> clazz = SML_Parameter_Change.class;
		String url = clazz.getResource("/" + clazz.getName().replace('.', '/') + ".class").toString();
		String pluginsDir = url.substring("file:".length(), url.length() - clazz.getName().length() - ".class".length());
		System.setProperty("plugins.dir", pluginsDir);		
		IJ.runPlugIn(clazz.getName(), "");
	}






	public void run(String arg) { 		

		String args =  ij.Macro.getOptions(); // get macro input.
		if (args != null)
		{			
			args = args.replaceAll("\\s+","");		// remove spaces, line separators
			String[] inputParameters = args.split(",");	// split based on ","
			GetParameters parameters = new GetParameters();
			parameters.get(); // load parameters.	
			parameters.set(inputParameters);
		}
	}

}
