import ij.IJ;
import ij.plugin.PlugIn;

public class SML_ParameterList_ implements PlugIn
{

	public static void main(final String... args) throws Exception {
		// create the ImageJ application context with all available services				
		final ij.ImageJ ij = new ij.ImageJ();
		ij.setVisible(true);

		
		Class<?> clazz = SML_ParameterList_.class;
		String url = clazz.getResource("/" + clazz.getName().replace('.', '/') + ".class").toString();
		String pluginsDir = url.substring("file:".length(), url.length() - clazz.getName().length() - ".class".length());
		System.setProperty("plugins.dir", pluginsDir);		
		IJ.runPlugIn(clazz.getName(), "");
		}

	public void run(String arg) {
		IJ.showMessage("SMLocalizer - v2.1.0",
				"Most parameters can be changed through the macro interface using the call 'run('SML Parameter Change',args);'. \n"
				+ "The list of parameters availible to be set through the macro interface changing args to the appropriate: \n \n"
				+ "Variable name \b \b\b\b\b\b\b args \n \n"
				+ "Input pixel size [nm]: 	\b 'inputpixelsize,VALUE'\n"
				+ "Total gain: 			\b\b\b\b\b\b\b\b\b\b	'totalgain,VALUE,CHANNEL(1-10)'\n"
				+ "Minimal signal: 		\b\b\b\b\b\b 'minimalsignal,VALUE,CHANNEL(1-10),STATUS(0 or 1)\n"
				+ "Filter width: 		\b\b\b\b\b\b\b\b\b 'filterwidth,VALUE,CHANNEL(1-10),STATUS(0 or 1)\n"
				+ "Photon count: 		\b\b\b\b\b\b\b	'photoncount,LOW_VALUE,HIGH_VALUE,CHANNEL(1-10),STATUS(0 or 1)'\n"
				+ "Sigma xy[nm]: 		\b\b\b\b\b\b  'sigmaxy,LOW_VALUE,HIGH_VALUE,CHANNEL(1-10),STATUS(0 or 1)'\n"
				+ "R^2:	 			\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b 'rsquare,LOW_VALUE,HIGH_VALUE,CHANNEL(1-10),STATUS(0 or 1)'\n"
				+ "Precision xy[nm]:	\b\b\b\b\b	'precisionxy,LOW_VALUE,HIGH_VALUE,CHANNEL(1-10),STATUS(0 or 1)'\n"
				+ "Precision z[nm]:		\b\b\b\b\b\b	'precisionz,LOW_VALUE,HIGH_VALUE,CHANNEL(1-10),STATUS(0 or 1)'\n"
				+ "Frame: 			\b\b\b\b\b\b\b\b\b\b\b\b\b		'frame,LOW_VALUE,HIGH_VALUE,CHANNEL(1-10),STATUS(0 or 1)'\n"
				+ "z[nm]: 		\b\b\b\b\b\b\b\b\b\b\b\b\b\b 'z,LOW_VALUE,HIGH_VALUE,CHANNEL(1-10),STATUS(0 or 1)'\n"
				+ "Drift Correct: 	\b\b\b\b\b\b\b\b		'driftcorrect,VALUE_XY,VALUE_Z,VALUE_BINS,CHANNEL(1-10),STATUS(0 or 1)'\n"
				+ "Channel align: 	\b\b\b\b\b\b\b	'channelalign,VALUE_XY,VALUE_Z,CHANNEL(1-10),STATUS(0 or 1)'\n"
				+ "Render image: 	\b\b\b\b\b\b 'renderimage,VALUE_XY,VALUE_Z,GAUSSIAN_SMOOTHING(0 or 1),STATUS(0 or 1)'\n \n"
				+ "Settings edited in this manner will applied to the most recently used settings set and new values will overwrite current values.");		
	}		
}
