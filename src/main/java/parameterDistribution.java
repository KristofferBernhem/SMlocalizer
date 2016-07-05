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
		double[] SigmaY = new double[ParticleList.size()];
		double[] PrecisionX = new double[ParticleList.size()];
		double[] PrecisionY = new double[ParticleList.size()];
		double[] CbiSquare = new double[ParticleList.size()];
		double[] Photons = new double[ParticleList.size()];
		for (int i = 0; i < SigmaX.length; i++){
			SigmaX[i] = ParticleList.get(i).sigma_x;
			SigmaY[i] = ParticleList.get(i).sigma_y;
			PrecisionX[i] = ParticleList.get(i).precision_x;
			PrecisionY[i] = ParticleList.get(i).precision_y;
			CbiSquare[i] = ParticleList.get(i).chi_square;
			Photons[i] = ParticleList.get(i).photons;
			
		}
		BackgroundCorrection.quickSort(SigmaX, 0, SigmaX.length-1);
		BackgroundCorrection.quickSort(SigmaY, 0, SigmaY.length-1);
		BackgroundCorrection.quickSort(PrecisionX, 0, SigmaY.length-1);
		BackgroundCorrection.quickSort(PrecisionY, 0, SigmaY.length-1);
		BackgroundCorrection.quickSort(CbiSquare, 0, SigmaY.length-1);
		BackgroundCorrection.quickSort(Photons, 0, SigmaY.length-1);
		double SigmaXmin = SigmaX[0]/1.1;
		double SigmaXmax = SigmaX[SigmaX.length-1]*1.1;		
		double[] x_axis = correctDrift.interp(SigmaXmin, SigmaXmax, nBins);
		double[] binnedDataX = binData(SigmaX,nBins,x_axis);
		double[] binnedDataY = binData(SigmaY,nBins,x_axis);
		String[] headerSigma = {"Sigma", "sigma [nm]", "count"};
		plot(headerSigma,"Sigma_x \n Sigma_y",binnedDataX,binnedDataY,x_axis);
		
		
		double PrecisionMin = PrecisionX[0]/1.1;
		double PrecisionMax = PrecisionX[PrecisionX.length-1]*1.1;
		x_axis = correctDrift.interp(PrecisionMin, PrecisionMax, nBins);
		String[] headerPrecision = {"Precision", "precision [nm]", "count"};
		plot(headerPrecision,"Precision_x \n Precision_y",binData(PrecisionX,nBins,x_axis),binData(PrecisionY,nBins,x_axis),correctDrift.interp(PrecisionMin, PrecisionMax, nBins));
		
		
		double ChiSquareMin = CbiSquare[0]/1.1;
		double ChiSquareMax = CbiSquare[CbiSquare.length-1]*1.1;
		x_axis = correctDrift.interp(ChiSquareMin, ChiSquareMax, nBins);
		String[] headerChiSquare = {"Chi^2", "Chi^2", "count"};
		plot(headerChiSquare,"Chi^2",binData(CbiSquare,nBins,x_axis),correctDrift.interp(ChiSquareMin, ChiSquareMax, nBins));			
		
		double PhotonsMin = Photons[0]/1.1;
		double PhotonsMax = Photons[Photons.length-1]*1.1;
		x_axis = correctDrift.interp(PhotonsMin, PhotonsMax, nBins);
		String[] headerSPhotons = {"Photons", "photons", "count"};
		plot(headerSPhotons,"Photons",binData(Photons,nBins,x_axis),correctDrift.interp(PhotonsMin, PhotonsMax, nBins));
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
	
	static void plot(String[] header, String Legend, double[] values,double[] x_axis) {
		//double[] x = new double[values.length];
		Plot newPlot = new Plot(header[0],header[1],header[2]);
		newPlot.addPoints(x_axis,values, Plot.LINE);
		newPlot.addLegend(Legend);
		newPlot.show();
		
	}
	static void plot(String[] header, String Legend, double[] values, double[] values2,double[] x_axis) {
		Plot newPlot = new Plot(header[0],header[1],header[2]);
		newPlot.setColor(Color.GREEN);
		newPlot.addPoints(x_axis,values, Plot.LINE);
		newPlot.setColor(Color.RED);
		newPlot.addPoints(x_axis,values2, Plot.LINE);
		newPlot.addLegend(Legend);
		newPlot.show();
	}
}
