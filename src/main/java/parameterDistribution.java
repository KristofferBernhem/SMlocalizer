import java.awt.Color;
import java.util.ArrayList;

import ij.IJ;
import net.imagej.ImageJ;
import ij.gui.Plot;
import ij.plugin.PlugIn;

public class parameterDistribution implements PlugIn{

	public static void main(String[] args){
		final ImageJ ij = new ImageJ();
		ij.ui().showUI();

		//	
		//showHistogram(ImPlus,10);

		Class<?> clazz = parameterDistribution.class;
		String url = clazz.getResource("/" + clazz.getName().replace('.', '/') + ".class").toString();
		String pluginsDir = url.substring("file:".length(), url.length() - clazz.getName().length() - ".class".length());
		System.setProperty("plugins.dir", pluginsDir);
		IJ.runPlugIn(clazz.getName(), "");
	}


	
	public void run(String arg0) {
		int nBins = 50;
		
		/*
		 * testdata:
		 */
		int nFrame =100;
		int width = 64;
		int height = 64;
		int perFrame = 10;
		int total = 100;
		ArrayList<Particle> ParticleList = TestData.generate(nFrame, width, height, perFrame, total);
		
		/*
		 * write for all parameters of interest. Multiwindow plotting? two color for xy
		 */
		
		double[] SigmaX = new double[ParticleList.size()];
		for (int i = 0; i < SigmaX.length; i++){
			SigmaX[i] = ParticleList.get(i).sigma_x;
		}
		BackgroundCorrection.quickSort(SigmaX, 0, SigmaX.length-1);
		double SigmaXmin = SigmaX[0];
		double SigmaXmax = SigmaX[SigmaX.length-1];
		
		double[] x_axis = correctDrift.interp(SigmaXmin, SigmaXmax, nBins);
		double[] binnedData = binData(SigmaX,nBins,x_axis);
		
		plot(binnedData,x_axis);
	}
	
	static double[] binData(double[] sortedData, int nBins,double[] x_axis){
		double[] binnedData = new double[nBins];
				
		int currBin = 1;
		for (int i = 0; i < sortedData.length; i++){
			if (sortedData[i] <= x_axis[currBin-1]){
				binnedData[currBin-1]++;
			}else{
				currBin++;
				binnedData[currBin-1]++;
			}							 
		}
		return binnedData;
		
	}
	
	static void plot(double[] values,double[] x_axis) {
		double[] x = new double[values.length];
		for (int i = 0; i < x.length; i++)
			x[i] = x_axis[i];
		Plot plot = new Plot("Plot window", "x", "values", x, values);
		plot.draw();
		plot.show();
	}
	static void plot(double[] values, double[] values2,int[] x_axis) {
		double[] x = new double[values.length];
		for (int i = 0; i < x.length; i++)
			x[i] = x_axis[i];
		Plot plot = new Plot("Plot window", "x", "values", x, values);
		plot.setColor(Color.GREEN);
		plot.draw();
		plot.addPoints(x, values, Plot.LINE);

		plot.setColor(Color.RED);
		plot.draw();
		plot.addPoints(x, values2, Plot.LINE);

		plot.addLegend("X: green" + "\n" + "Y: red");
		plot.show();
	}
}
