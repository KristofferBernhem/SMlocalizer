import java.awt.Color;
import java.util.ArrayList;
import ij.gui.Plot;


public class parameterDistribution{

/*	public static void main(String[] args){
		final ImageJ ij = new ImageJ();
		ij.ui().showUI();

		//	
		//showHistogram(ImPlus,10);

		Class<?> clazz = parameterDistribution.class;
		String url = clazz.getResource("/" + clazz.getName().replace('.', '/') + ".class").toString();
		String pluginsDir = url.substring("file:".length(), url.length() - clazz.getName().length() - ".class".length());
		System.setProperty("plugins.dir", pluginsDir);

		/*
		 * testdata:
		 */
/*		int nFrame =100;
		int width = 64;
		int height = 64;
		int perFrame = 10;
		int total = 100;
		ArrayList<Particle> ParticleList = TestData.generate(nFrame, width, height, perFrame, total);
		TableIO.Store(ParticleList);
		IJ.runPlugIn(clazz.getName(), "");
	} */



	public static void run() {
		int nBins = 100;
		ArrayList<Particle> ParticleList = TableIO.Load(); // Get data.


		int Channel = 1;
		for (int i = 0; i < ParticleList.size(); i++){
			if ( ParticleList.get(i).channel > Channel){
				Channel = (int) ParticleList.get(i).channel;
			}
		}
		for (int Ch = 1; Ch <= Channel; Ch++){ // Remake based on logic from drift correction binning.
			double[] tempSigmaX = new double[ParticleList.size()];
			double[] tempSigmaY = new double[ParticleList.size()];
			double[] tempSigmaZ = new double[ParticleList.size()];
			double[] tempPrecisionX = new double[ParticleList.size()];
			double[] tempPrecisionY = new double[ParticleList.size()];
			double[] tempPrecisionZ = new double[ParticleList.size()];
			double[] tempCbiSquare = new double[ParticleList.size()];
			double[] tempPhotons = new double[ParticleList.size()];
			int count = 0;
			for (int i = 0; i < ParticleList.size(); i++){
				if(ParticleList.get(i).channel == Ch){
					tempSigmaX[i] 		= ParticleList.get(i).sigma_x;
					tempSigmaY[i] 		= ParticleList.get(i).sigma_y;
					tempSigmaZ[i] 		= ParticleList.get(i).sigma_z;
					tempPrecisionX[i] 	= ParticleList.get(i).precision_x;
					tempPrecisionY[i] 	= ParticleList.get(i).precision_y;
					tempPrecisionZ[i] 	= ParticleList.get(i).precision_z;
					tempCbiSquare[i] 	= ParticleList.get(i).chi_square;
					tempPhotons[i] 		= ParticleList.get(i).photons;
					count++;
				}
			}
			
			double[] SigmaX = new double[count];
			double[] SigmaY = new double[count];
			double[] SigmaZ = new double[count];
			double[] PrecisionX = new double[count];
			double[] PrecisionY = new double[count];
			double[] PrecisionZ = new double[count];
			double[] CbiSquare = new double[count];
			double[] Photons = new double[count];
			for(int i = 0; i < count; i++){
				SigmaX[i] 		= tempSigmaX[i];
				SigmaY[i] 		= tempSigmaY[i];
				SigmaZ[i] 		= tempSigmaZ[i];
				PrecisionX[i] 	= tempPrecisionX[i];
				PrecisionY[i] 	= tempPrecisionY[i];
				PrecisionZ[i] 	= tempPrecisionZ[i];
				CbiSquare[i] 	= tempCbiSquare[i];
				Photons[i] 		= tempPhotons[i];
			}
			
			BackgroundCorrection.quickSort(SigmaX, 0, SigmaX.length-1);
			BackgroundCorrection.quickSort(SigmaY, 0, SigmaY.length-1);
			BackgroundCorrection.quickSort(SigmaZ, 0, SigmaZ.length-1);
			BackgroundCorrection.quickSort(PrecisionX, 0, PrecisionX.length-1);
			BackgroundCorrection.quickSort(PrecisionY, 0, PrecisionY.length-1);
			BackgroundCorrection.quickSort(PrecisionZ, 0, PrecisionZ.length-1);
			BackgroundCorrection.quickSort(CbiSquare, 0, CbiSquare.length-1);
			BackgroundCorrection.quickSort(Photons, 0, Photons.length-1);
			double SigmaMin = SigmaX[0]/1.1;
			double SigmaMax = SigmaX[SigmaX.length-1]*1.1;
			if (SigmaY[0]/1.1 < SigmaMin)
				SigmaMin = SigmaY[0]/1.1;
			if (SigmaY[SigmaY.length-1]*1.1 > SigmaMax)
				SigmaMax = SigmaY[SigmaY.length-1]*1.1;

			double[] x_axis = correctDrift.interp(SigmaMin, SigmaMax, nBins);
			double[] binnedDataX = binData(SigmaX,nBins,x_axis);
			double[] binnedDataY = binData(SigmaY,nBins,x_axis);
			String[] headerSigma = {("Channel " + Ch +" sigma"), "sigma [nm]", "count"};
			plot(headerSigma,"Sigma_x \n Sigma_y",binnedDataX,binnedDataY,x_axis);
			
			double SigmaZmin = SigmaZ[0]/1.1;
			double SigmaZmax = SigmaZ[SigmaZ.length-1]*1.1;
			x_axis = correctDrift.interp(SigmaZmin, SigmaZmax, nBins);
			String[] headerSigmaZ = {("Channel " + Ch +" sigma z"), "sigma z [nm]", "count"};
			plot(headerSigmaZ,"Sigma_z",binData(SigmaZ,nBins,x_axis),correctDrift.interp(SigmaZmin, SigmaZmax, nBins));			

			double PrecisionMin = PrecisionX[0]/1.1;
			double PrecisionMax = PrecisionX[PrecisionX.length-1]*1.1;
			if (PrecisionY[0]/1.1 < PrecisionMin)
				PrecisionMin = PrecisionY[0]/1.1;
			if (PrecisionY[PrecisionY.length-1]*1.1 > PrecisionMax)
				PrecisionMax = PrecisionY[PrecisionY.length-1]*1.1;

			x_axis = correctDrift.interp(PrecisionMin, PrecisionMax, nBins);
			String[] headerPrecision = {("Channel " + Ch +" Precision"), "precision [nm]", "count"};
			plot(headerPrecision,"Precision_x \n Precision_y",binData(PrecisionX,nBins,x_axis),binData(PrecisionY,nBins,x_axis),correctDrift.interp(PrecisionMin, PrecisionMax, nBins));		
			
			double PrecisionZmin = PrecisionZ[0]/1.1;
			double PrecisionZmax = PrecisionZ[PrecisionZ.length-1]*1.1;
			x_axis = correctDrift.interp(PrecisionZmin, PrecisionZmax, nBins);
			String[] headerPrecisionZ = {("Channel " + Ch +" Chi^2"), "Chi^2", "count"};
			plot(headerPrecisionZ,"Precision_z",binData(PrecisionZ,nBins,x_axis),correctDrift.interp(PrecisionZmin, PrecisionZmax, nBins));			

			double ChiSquareMin = CbiSquare[0]/1.1;
			double ChiSquareMax = CbiSquare[CbiSquare.length-1]*1.1;
			x_axis = correctDrift.interp(ChiSquareMin, ChiSquareMax, nBins);
			String[] headerChiSquare = {("Channel " + Ch +" Chi^2"), "Chi^2", "count"};
			plot(headerChiSquare,"Chi^2",binData(CbiSquare,nBins,x_axis),correctDrift.interp(ChiSquareMin, ChiSquareMax, nBins));			

			double PhotonsMin = Photons[0]/1.1;
			double PhotonsMax = Photons[Photons.length-1]*1.1;
			x_axis = correctDrift.interp(PhotonsMin, PhotonsMax, nBins);
			String[] headerSPhotons = {("Channel " + Ch +" 	aPhotons"), "photons", "count"};
			plot(headerSPhotons,"Photons",binData(Photons,nBins,x_axis),correctDrift.interp(PhotonsMin, PhotonsMax, nBins));
		} // End loop over channels.
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
