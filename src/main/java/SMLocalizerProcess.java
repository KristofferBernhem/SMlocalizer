
import ij.WindowManager;

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
/**
*
* @author kristoffer.bernhem@gmail.com
*/

/*
 * TODO: Add storage of gaussfilter setting.
 * TODO Re implemeent functions. 
 * 
 */
public class SMLocalizerProcess {

	public static void main(final String... args) throws Exception {
		String storeName =ij.Prefs.get("SMLocalizer.CurrentSetting", "");
		String outName = (ij.Prefs.get("SMLocalizer.settings."+storeName+
				".pixelSizeZ", 
				""));
		System.out.println(outName + " c " + storeName);
	}

	public static void execute(boolean CPU) {
		/*
		 * Load user settings and run process function.		 
		 */

		/************************************************************************
		 ************************ LOAD USER SETTINGS ****************************
		 ************************************************************************/

		String storeName = ij.Prefs.get("SMLocalizer.CurrentSetting", "");

		/*
		 * non channel unique variables.
		 */
		boolean doDfriftCorr 	= false;
		boolean doChAlign 		= false;
		int[] pixelSize 		= new int[2];
		boolean doGaussianSmoothing = true;
		if (ij.Prefs.get("SMLocalizer.settings."+storeName+
				".doDriftCorrect.",1) == 1)
			doDfriftCorr = true;		

		if (ij.Prefs.get("SMLocalizer.settings."+storeName+
				".doChannelAlign.",1) == 1)
			doChAlign = true;		

		// pixel size XY
		pixelSize[0] = Integer.parseInt(
				ij.Prefs.get("SMLocalizer.settings."+storeName+
						".pixelSize", 
						""));
		// pixel size Z
		pixelSize[1] = Integer.parseInt(
				ij.Prefs.get("SMLocalizer.settings."+storeName+
						".pixelSizeZ", 
						""));
		/*
		 * Basic input
		 */
		int[] totalGain	 			= new int[10];
		int[] minPixelOverBkgrnd 	= new int[10];
		int[] signalStrength 		= new int[10];
		int[] gWindow 				= new int[10];
		int[] window 				= new int[10];
		double[] minDistance 		= new double[10];

		/*
		 * Cluster analysis
		 */
		boolean[] doClusterAnalysis = new boolean[10];
		double[] epsilon 			= new double[10];
		int[] minPtsCluster 		= new int[10];

		/*
		 * Render image settings
		 */		
		boolean[] doRender 			= new boolean[10];

		/*
		 * Parameter settings
		 */
		boolean[][] include = new boolean[7][10];
		double[][] lb = new double[7][10];
		double[][] ub = new double[7][10];

		/*
		 * Drift correction and channel alignment
		 */
		int[] driftCorrLowCount 	= new int[10]; 
		int[] driftCorrHighCount 	= new int[10];
		int[][] driftCorrshift 		= new int[2][10];
		int[] driftCorrBins 		= new int[10];
		int[] chAlignLowCount 		= new int[10];
		int[] chAlignHighCount 		= new int[10];
		int[][] chAlignshift 		= new int[2][10];


		for (int Ch = 0; Ch < 10; Ch++){
			/*
			 *   Basic input settings
			 */	    		                           		   	
			// total gain
			totalGain[Ch] = Integer.parseInt(
					ij.Prefs.get("SMLocalizer.settings."+storeName+
							".totaGain."+Ch, 
							""));		   
			// minimum pixel over background
			minPixelOverBkgrnd[Ch] = Integer.parseInt(
					ij.Prefs.get("SMLocalizer.settings."+storeName+
							".minPixelOverBackground."+Ch, 
							""));		   
			// minimal signal
			signalStrength[Ch] = Integer.parseInt(
					ij.Prefs.get("SMLocalizer.settings."+storeName+
							".minimalSignal."+Ch, 
							""));
			// gauss window size
			gWindow[Ch] = Integer.parseInt( 
					ij.Prefs.get("SMLocalizer.settings."+storeName+
							".gaussWindow."+Ch, 
							""));
			// gauss window size
			window[Ch] = Integer.parseInt(
					ij.Prefs.get("SMLocalizer.settings."+storeName+
							".windowWidth."+Ch, 
							""));
			// min pixel distance
			minDistance[Ch] = Double.parseDouble(			
					ij.Prefs.get("SMLocalizer.settings."+storeName+
							".minPixelDist."+Ch, 
							""));
			/*
			 *       Cluster analysis settings
			 */
			// perform cluster analysis
			if (ij.Prefs.get("SMLocalizer.settings."+storeName+
					".doClusterAnalysis."+Ch, 
					"").equals("1"))
				doClusterAnalysis[Ch] = true;
			else
				doClusterAnalysis[Ch] = false;

			// min pixel distance	
			epsilon[Ch] = Double.parseDouble(
					ij.Prefs.get("SMLocalizer.settings."+storeName+
							".epsilon."+Ch, 
							""));
			// min pixel distance
			minPtsCluster[Ch] = Integer.parseInt(			 
					ij.Prefs.get("SMLocalizer.settings."+storeName+
							".minPtsCluster."+Ch, 
							""));  		    
			/*
			 *       Render image settings.
			 */
			// render image
			if (ij.Prefs.get("SMLocalizer.settings."+storeName+
					".doRenderImage."+Ch, 
					"").equals("1"))
				doRender[Ch] = true;
			else
				doRender[Ch] = false;		

			/*
			 * store parameter settings:
			 */

			// photon count

			if (ij.Prefs.get("SMLocalizer.settings."+storeName+
					".doPotonCount."+Ch, 
					"").equals("1"))
				include[0][Ch] = true;
			else
				include[0][Ch] = false;

			lb[0][Ch] = Double.parseDouble( 
					ij.Prefs.get("SMLocalizer.settings."+storeName+
							".minPotonCount."+Ch, 
							""));  
			ub[0][Ch] = Double.parseDouble( 
					ij.Prefs.get("SMLocalizer.settings."+storeName+
							".maxPotonCount."+Ch, 
							""));  		    
			// Sigma XY  
			if (ij.Prefs.get("SMLocalizer.settings."+storeName+
					".doSigmaXY."+Ch, 
					"").equals("1"))
				include[1][Ch] = true;
			else
				include[1][Ch] = false;

			lb[1][Ch] = Double.parseDouble( 
					ij.Prefs.get("SMLocalizer.settings."+storeName+
							".minSigmaXY."+Ch, 
							""));  
			ub[1][Ch] = Double.parseDouble( 
					ij.Prefs.get("SMLocalizer.settings."+storeName+
							".maxSigmaXY."+Ch, 
							"")); 

			// Sigma Z 		    
			if (ij.Prefs.get("SMLocalizer.settings."+storeName+
					".doSigmaZ."+Ch, 
					"").equals("1"))
				include[2][Ch] = true;
			else
				include[2][Ch] = false;

			lb[2][Ch] = Double.parseDouble( 
					ij.Prefs.get("SMLocalizer.settings."+storeName+
							".minSigmaZ."+Ch, 
							""));  
			ub[2][Ch] = Double.parseDouble( 
					ij.Prefs.get("SMLocalizer.settings."+storeName+
							".maxSigmaZ."+Ch, 
							"")); 
			// Rsquare  
			if (ij.Prefs.get("SMLocalizer.settings."+storeName+
					".doRsquare."+Ch, 
					"").equals("1"))
				include[3][Ch] = true;
			else
				include[3][Ch] = false;

			lb[3][Ch] = Double.parseDouble( 
					ij.Prefs.get("SMLocalizer.settings."+storeName+
							".minRsquare."+Ch, 
							""));  
			ub[3][Ch] = Double.parseDouble( 
					ij.Prefs.get("SMLocalizer.settings."+storeName+
							".maxRsquare."+Ch, 
							"")); 

			// Precision XY
			if (ij.Prefs.get("SMLocalizer.settings."+storeName+
					".doPrecisionXY."+Ch, 
					"").equals("1"))
				include[4][Ch] = true;
			else
				include[4][Ch] = false;

			lb[4][Ch] = Double.parseDouble( 
					ij.Prefs.get("SMLocalizer.settings."+storeName+
							".minPrecisionXY."+Ch, 
							""));  
			ub[4][Ch] = Double.parseDouble( 
					ij.Prefs.get("SMLocalizer.settings."+storeName+
							".maxPrecisionXY."+Ch, 
							""));

			// Precision Z
			if (ij.Prefs.get("SMLocalizer.settings."+storeName+
					".doPrecisionZ."+Ch, 
					"").equals("1"))
				include[5][Ch] = true;
			else
				include[5][Ch] = false;

			lb[5][Ch] = Double.parseDouble(  
					ij.Prefs.get("SMLocalizer.settings."+storeName+
							".minPrecisionZ."+Ch, 
							""));  
			ub[5][Ch] = Double.parseDouble( 
					ij.Prefs.get("SMLocalizer.settings."+storeName+
							".maxPrecisionZ."+Ch, 
							""));

			// Frame
			if (ij.Prefs.get("SMLocalizer.settings."+storeName+
					".doFrame."+Ch, 
					"").equals("1"))
				include[6][Ch] = true;
			else
				include[6][Ch] = false;

			lb[6][Ch] = Double.parseDouble( 
					ij.Prefs.get("SMLocalizer.settings."+storeName+
							".minFrame."+Ch, 
							""));  
			ub[6][Ch] = Double.parseDouble(  
					ij.Prefs.get("SMLocalizer.settings."+storeName+
							".maxFrame."+Ch, 
							""));

			/*
			 *   Drift and channel correct settings
			 */

			// drift correction bins.
			driftCorrLowCount[Ch] = Integer.parseInt( 
					ij.Prefs.get("SMLocalizer.settings."+storeName+
							".driftCorrBinLow."+Ch, 
							""));  
			driftCorrHighCount[Ch] = Integer.parseInt( 
					ij.Prefs.get("SMLocalizer.settings."+storeName+
							".driftCorrBinHigh."+Ch,
							""));

			// drift correction shift
			driftCorrshift[0][Ch] = Integer.parseInt( 
					ij.Prefs.get("SMLocalizer.settings."+storeName+
							".driftCorrShiftXY."+Ch, 
							""));  
			driftCorrshift[1][Ch] = Integer.parseInt( 
					ij.Prefs.get("SMLocalizer.settings."+storeName+
							".driftCorrShiftZ."+Ch,
							""));			    
			// number of drift bins
			driftCorrBins[Ch] = Integer.parseInt( 
					ij.Prefs.get("SMLocalizer.settings."+storeName+
							".driftCorrBin."+Ch,
							""));	


			// channel align bin low
			chAlignLowCount[Ch] = Integer.parseInt( 
					ij.Prefs.get("SMLocalizer.settings."+storeName+
							".chAlignBinLow."+Ch,
							""));	

			// channel align bin high
			chAlignHighCount[Ch] = Integer.parseInt( 
					ij.Prefs.get("SMLocalizer.settings."+storeName+
							".chAlignBinHigh."+Ch,
							""));	


			// channel align shift
			chAlignshift[0][Ch] = Integer.parseInt( 
					ij.Prefs.get("SMLocalizer.settings."+storeName+
							".chAlignShiftXY."+Ch,
							""));	
			chAlignshift[1][Ch] = Integer.parseInt( 
					ij.Prefs.get("SMLocalizer.settings."+storeName+
							".chAlignShiftZ."+Ch,
							""));	 

		} // end Ch loop for variable loading.
		
	
	//	int selectedModel = 0;
		if (CPU) // CPU processing
		{
			BackgroundCorrection.medianFiltering(window,WindowManager.getCurrentImage(),0); // correct background.
			ij.IJ.log("background ok");
//			double maxSigma = 2; // 2D
//			localizeAndFit.run(signalStrength, gWindow, pixelSize,minPixelOverBkgrnd,totalGain,selectedModel,maxSigma);  //locate and fit all particles.
			ij.IJ.log("localize ok");
		} // end CPU processing
		else // GPU processing 
		{
	//		selectedModel = 2; // tell functions to utilize GPU.
//			processMedianFit.run(window, WindowManager.getCurrentImage(), signalStrength, gWindow, pixelSize, minPixelOverBkgrnd, totalGain); // GPU specific call.
		} // end GPU processing
		cleanParticleList.run(lb,ub,include);
		ij.IJ.log("clean list ok ok");
		
		if (doDfriftCorr)
		{	         
		//	correctDrift.run(driftCorrshift, driftCorrBins, driftCorrHighCount, driftCorrLowCount, selectedModel,pixelSize); // drift correct all channels.
		}
		if (doChAlign)
		{
	//		correctDrift.ChannelAlign(chAlignshift, chAlignHighCount, chAlignLowCount,selectedModel); // drift correct all channels.
		}
		boolean doCluster = false;
		for (int ch = 0; ch < 10; ch++){ // if any channels require cluster analysis, perform it.
			if (doClusterAnalysis[ch])
				doCluster = true;
		}
		if (doCluster)
			DBClust.Ident(epsilon, minPtsCluster,pixelSize,doClusterAnalysis); // change call to include no loop but checks for number of channels within DBClust.	

		RenderIm.run(doRender,pixelSize,doGaussianSmoothing); // Not 3D yet

	}

}
