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
import ij.WindowManager;
import ij.plugin.PlugIn;

public class SML_Process implements PlugIn {

	public static void main(final String... args) throws Exception {
		// create the ImageJ application context with all available services				
		final ij.ImageJ ij = new ij.ImageJ();
		ij.setVisible(true);

		
		Class<?> clazz = SML_Process.class;
		String url = clazz.getResource("/" + clazz.getName().replace('.', '/') + ".class").toString();
		String pluginsDir = url.substring("file:".length(), url.length() - clazz.getName().length() - ".class".length());
		System.setProperty("plugins.dir", pluginsDir);		
		IJ.runPlugIn(clazz.getName(), "");
		}


		
	
	
	
	public void run(String arg) { // macro input arg should be comma separated with window length and computation mode.		
		/*	Macro:	
		 * arg = "0,2D,5,15";  CPU (0) /GPU (2), modality.
		 * run("SML Process",arg);
		 */	
		String args =  ij.Macro.getOptions(); // get macro input.
		int selectedModel 		= 0;
		String modalityChoice 	= "2D";		
		int[]outputPixelSize = new int[2];
		GetParameters parameters = new GetParameters();
		parameters.get(); // load parameters.
		if (args != null)
		{
			args = args.replaceAll("\\s+","");		// remove spaces, line separators
			String[] inputParameters = args.split(",");	// split based on ","
			selectedModel 	= Integer.parseInt(inputParameters[0]);
			modalityChoice 	= inputParameters[1];
			if (inputParameters.length > 2)
				{
					outputPixelSize[0] = Integer.parseInt(inputParameters[2]);
					if (inputParameters.length > 3)
					{
						outputPixelSize[1] = Integer.parseInt(inputParameters[3]);
					}
					else
					{
						outputPixelSize[1] = parameters.outputPixelSize[1];						
					}

				}else
				{
					outputPixelSize[0] = parameters.outputPixelSize[0];
					outputPixelSize[1] = parameters.outputPixelSize[1];
				}
			
					
		}else
		{
			//Custom button text
			Object[] options = {"CPU",
					"GPU",
			"Cancel"};
			Object[] possibilities = {"2D", "PRILM", "Biplane","Double Helix", "Astigmatism"};
			int n = JOptionPane.showOptionDialog(null,
					"Select processing method",
					"Processing",
					JOptionPane.YES_NO_CANCEL_OPTION,
					JOptionPane.PLAIN_MESSAGE,
					null,
					options,			    
					options[0]);			

			switch (n)
			{
			case 0: 	selectedModel = 0; break;
			case 1: 	selectedModel = 2; break;
			case 2: 	selectedModel = 5; break;
			}

			if (selectedModel != 5)
			{
				String s = (String)JOptionPane.showInputDialog(
						null,
						"Select modality used",
						"Imaging modality",
						JOptionPane.PLAIN_MESSAGE,
						null,
						possibilities,
						possibilities[0]);
				if (s != null)
				{
					s = s.replaceAll("\\s+","");		// remove spaces, line separators
					modalityChoice = s;
				}
				else
				{
					modalityChoice = null;
				}
				if (modalityChoice != null)
				{
					s = (String)JOptionPane.showInputDialog(
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
				}
			}
		}

		if (modalityChoice != null && selectedModel != 5)
		{


			// Locate and fit all objects.
			/*
			 * First, initiate and populate all input variables with default values.
			 */

			int[] totalGain 		= parameters.totalGain;
			int gWindow 			= 5;
			int pixelSize 			= parameters.pixelSize;

			int[] signalStrength 	= new int[10];
			double maxSigma 		= 2; // 2D 




			if (modalityChoice.equalsIgnoreCase("2D"))
			{
				if (pixelSize < 100) // if smaller pixel size the window width needs to increase.
				{
					gWindow = (int) Math.ceil(500 / pixelSize); // 500 nm wide window.

				}
				if (gWindow%2 == 0)
					gWindow++;			 
			} // 2D.
			else if(modalityChoice.equalsIgnoreCase("PRILM"))
			{
				gWindow 		= (int)ij.Prefs.get("SMLocalizer.calibration.PRILM.window", 0);
				maxSigma 		= (int)ij.Prefs.get("SMLocalizer.calibration.PRILM.sigma", 0);
			}
			else if(modalityChoice.equalsIgnoreCase("Biplane"))
			{
				gWindow 		= (int)ij.Prefs.get("SMLocalizer.calibration.Biplane.window", 0);
				maxSigma 		= (int)ij.Prefs.get("SMLocalizer.calibration.Biplane.sigma", 0);
			}
			else if(modalityChoice.equalsIgnoreCase("DoubleHelix"))
			{
				gWindow 		= (int)ij.Prefs.get("SMLocalizer.calibration.DoubleHelix.window", 0);
				maxSigma 		= (int)ij.Prefs.get("SMLocalizer.calibration.DoubleHelix.sigma", 0);
			}
			else if(modalityChoice.equalsIgnoreCase("Astigmatism"))
			{
				gWindow 		= (int)ij.Prefs.get("SMLocalizer.calibration.Astigmatism.window", 0);
				maxSigma 		= (int)ij.Prefs.get("SMLocalizer.calibration.Astigmatism.sigma", 0);			 			
			}				
			for (int i = 0; i < 10; i++)
			{
				if(parameters.doMinimalSignalChList[i])
				{
					signalStrength[i] = parameters.minSignal[i];

				}else
				{
					signalStrength[i] = 0;
				}
			}
			if (selectedModel == 0)
			{
				BackgroundCorrection.medianFiltering(parameters.windowWidth,WindowManager.getCurrentImage(),selectedModel); // correct background.			 
				localizeAndFit.run(signalStrength, gWindow, pixelSize,  totalGain , selectedModel, maxSigma, modalityChoice);  //locate and fit all particles.
			}else if (selectedModel == 2)
			{
				processMedianFit.run(parameters.windowWidth, WindowManager.getCurrentImage(), signalStrength, pixelSize, totalGain, maxSigma, gWindow, modalityChoice); // GPU specific call. 
			}
			
			cleanParticleList.run(parameters.lb,parameters.ub,parameters.includeParameters); // clean out results table.			
			
			if (parameters.doDriftCorrect)
			{
				int[] bins         = parameters.numberOfBinsDriftCorr;
				int[][] boundry    = parameters.driftCorrShift;
				correctDrift.run(boundry, bins, selectedModel); // drift correct all channels.
			}
			if (parameters.doChannelAlign)
			{

				int[][] boundry = parameters.chAlignShift;
				correctDrift.ChannelAlign(boundry, selectedModel); // drift correct all channels.
			}
			
			boolean[] doCluster = parameters.doClusterAnalysis;
			double[] epsilon     = parameters.epsilon;
			int[] minPts        = parameters.minPtsCluster;
			boolean doClusterAnalysis = false;
			for (int ch = 0; ch < 10; ch++){
				if (doCluster[ch])
					doClusterAnalysis = true;
			}
			
			if (doClusterAnalysis)
				DBClust.Ident(epsilon, minPts,outputPixelSize,doCluster); // change call to include no loop but checks for number of channels within DBClust.	
			
			RenderIm.run(parameters.doRender,outputPixelSize,parameters.doGaussianSmoothing); // Not 3D yet
		}
	}
}


