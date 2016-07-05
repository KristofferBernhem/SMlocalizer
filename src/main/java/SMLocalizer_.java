

import ij.IJ;
import ij.plugin.PlugIn;
import net.imagej.ImageJ;


public class SMLocalizer_ implements PlugIn {

	public static void main(final String... args) throws Exception {
		// create the ImageJ application context with all available services
		final ImageJ ij = new ImageJ();
		ij.ui().showUI();
		Class<?> clazz = SMLocalizer_.class;
		String url = clazz.getResource("/" + clazz.getName().replace('.', '/') + ".class").toString();
		String pluginsDir = url.substring("file:".length(), url.length() - clazz.getName().length() - ".class".length());
		System.setProperty("plugins.dir", pluginsDir);
		IJ.runPlugIn(clazz.getName(), "");
	}


	public void run(String arg0) {
		String args[] = null;
		SMLocalizerGUI.main(args); //call GUI.
	}

}