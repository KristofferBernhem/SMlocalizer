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
import java.util.ArrayList;

import javax.swing.DefaultComboBoxModel;
import javax.swing.JComboBox;
import javax.swing.JLabel;
import javax.swing.JOptionPane;
import javax.swing.JPanel;

import ij.WindowManager;


/*TODO x-mas 2016:
 * Implement 3D calibrations:
 * 		DONE Write new fit call to bypass regular localize/process calls.
 * 		DONE Implement chromatic offset correction in 3D fit algorithms 
 * 		PRILM: preliminary complete.
 * 		Biplane: preliminary complete.
 * 		Double helix: preliminary complete.
 * 		Astigmatism: preliminary complete.
 * DONE_ Implement chromatic offset correction in 2D fit algorithms
 * DONE: Calibrate push button implemented
 * DONE: ROI size based on pixelsize and modality
 * DONE: Set pixels over background based on ROI size.
 * DONE: Update GPU fitting code to handle 3D (new initial search and input)
 * Check errors in cluster analysis.
 * DONE: Add fiducial checkbox saving of choice. fiducialsChList
 * DONE: Add doCorrelativeChList for choice of channel alignment method.
 * DONE: Add doChromaticChList for choice of channel alignment method.
 * DONE: set default correlativeCorr and chromaticCorr
 * update tooltips (align channel incorrect)
 * Make fiducial track applied during initial fitting. Move fiducial checkbox to basic settings.
 * DONE: Remove channel independent pixel size.
 * DONE: (error loading default file). ERROR: Not loading values correctly in UBUNTU with java 1.8.0_101. Check with newer version.
 * DONE: Remove sigmaZ from parameter list in code.
 * 
 * 
 * 
 * 
 * BONUS:
 * GPU transfer function for speedup.
 * Possible multi-emitter fitting.
 * Transfer function to LAMA
 * 
 * Manuscript: 
 * Compare with ground truth
 * 
 * 
 */

/**
 *
 * @author kristoffer.bernhem@gmail.com
 */

@SuppressWarnings("serial")
 public class SMLocalizerGUI extends javax.swing.JFrame {

	 public SMLocalizerGUI() {
		 initComponents();


		 try{			
			 loadParameters(ij.Prefs.get("SMLocalizer.CurrentSetting", "")); // current.
		 }catch (NumberFormatException e)
		 {        	
			 outputPixelSize.setText("5");
			 outputPixelSizeZ.setText("10");
			 doChannelAlign.setSelected(false);
			 doDriftCorrect.setSelected(true);
			 doGaussianSmoothing.setSelected(false); 
			 for (int id = 0; id < 10; id++){ // update list of variables for each ch.
				 /*
				  *   Basic input settings
				  */
				 pixelSizeChList.getItem(id).setText("100");
				 totalGainChList.getItem(id).setText("100");                        
				 minimalSignalChList.getItem(id).setText("2000");
				 gaussWindowChList.getItem(id).setText("1");
				 windowWidthChList.getItem(id).setText("101");				
				 /*
				  *       Cluster analysis settings
				  */
				 doClusterAnalysisChList.getItem(id).setText("0");
				 epsilonChList.getItem(id).setText("10");
				 minPtsClusterChList.getItem(id).setText("5");

				 /*
				  *       Render image settings.
				  */
				 doRenderImageChList.getItem(id).setText("1");	

				 /*
				  *   Drift and channel correct settings
				  */
				 driftCorrBinLowCountChList.getItem(id).setText("500");
				 driftCorrBinHighCountChList.getItem(id).setText("1000");
				 driftCorrShiftXYChList.getItem(id).setText("100");
				 driftCorrShiftZChList.getItem(id).setText("100");
				 numberOfBinsDriftCorrChList.getItem(id).setText("25");
				 chAlignBinLowCountChList.getItem(id).setText("500");
				 chAlignBinHighCountChList.getItem(id).setText("1000");
				 chAlignShiftXYChList.getItem(id).setText("150");
				 chAlignShiftZChList.getItem(id).setText("150");
				 doCorrelativeChList.getItem(id).setText("0");
				 doChromaticChList.getItem(id).setText("0");
				 fiducialsChList.getItem(id).setText("0");

				 /*
				  *   Parameter settings
				  */
				 //Photon count
				 doPhotonCountChList.getItem(id).setText("0");
				 minPhotonCountChList.getItem(id).setText("100");
				 maxPhotonCountChList.getItem(id).setText("1000");
				 // Sigma XY        
				 doSigmaXYChList.getItem(id).setText("1");       
				 minSigmaXYChList.getItem(id).setText("100");
				 maxSigmaXYChList.getItem(id).setText("200");  
            
				 // Rsquare        
				 doRsquareChList.getItem(id).setText("1");       
				 minRsquareChList.getItem(id).setText("0.9");
				 maxRsquareChList.getItem(id).setText("1.0"); 
				 // Precision XY
				 doPrecisionXYChList.getItem(id).setText("1");       
				 minPrecisionXYChList.getItem(id).setText("5");
				 maxPrecisionXYChList.getItem(id).setText("50"); 
				 // Precision Z
				 doPrecisionZChList.getItem(id).setText("0");       
				 minPrecisionZChList.getItem(id).setText("5");
				 maxPrecisionZChList.getItem(id).setText("75"); 
				 // Frame
				 doFrameChList.getItem(id).setText("0");
				 minFrameChList.getItem(id).setText("1");
				 maxFrameChList.getItem(id).setText("100000");
			 }

			 /*
			  * 2D and 3D calibration defaults:
			  */
			 for (int i = 1; i < 10; i++)
				{
					ij.Prefs.set("SMLocalizer.calibration.2D.ChOffsetX"+i,0);
					ij.Prefs.set("SMLocalizer.calibration.2D.ChOffsetY"+i,0);
				}
				ij.Prefs.set("SMLocalizer.calibration.2D.channels",0);
				
				for (int i = 1; i < 10; i++)
				{
					ij.Prefs.set("SMLocalizer.calibration.PRILM.ChOffsetX"+i,0);
					ij.Prefs.set("SMLocalizer.calibration.PRILM.ChOffsetY"+i,0);
					ij.Prefs.set("SMLocalizer.calibration.PRILM.ChOffsetZ"+i,0);
				}
				ij.Prefs.set("SMLocalizer.calibration.PRILM.window",5);
				ij.Prefs.set("SMLocalizer.calibration.PRILM.sigma",3);		
				ij.Prefs.set("SMLocalizer.calibration.PRILM.height",0);
				ij.Prefs.set("SMLocalizer.calibration.PRILM.channels",1);
				ij.Prefs.set("SMLocalizer.calibration.PRILM.step",0);
				ij.Prefs.set("SMLocalizer.calibration.PRILM.Ch1.0",0);
				for (int i = 1; i < 10; i++)
				{
					ij.Prefs.set("SMLocalizer.calibration.DoubleHelix.ChOffsetX"+i,0);
					ij.Prefs.set("SMLocalizer.calibration.DoubleHelix.ChOffsetY"+i,0);
					ij.Prefs.set("SMLocalizer.calibration.DoubleHelix.ChOffsetZ"+i,0);
				}
				ij.Prefs.set("SMLocalizer.calibration.DoubleHelix.window",5);
				ij.Prefs.set("SMLocalizer.calibration.DoubleHelix.sigma",2);		
				ij.Prefs.set("SMLocalizer.calibration.DoubleHelix.height",0);
				ij.Prefs.set("SMLocalizer.calibration.DoubleHelix.channels",1);
				ij.Prefs.set("SMLocalizer.calibration.DoubleHelix.step",0);
				ij.Prefs.set("SMLocalizer.calibration.DoubleHelix.Ch1.0",0);
				for (int i = 1; i < 10; i++)
				{
					ij.Prefs.set("SMLocalizer.calibration.Biplane.ChOffsetX"+i,0);
					ij.Prefs.set("SMLocalizer.calibration.Biplane.ChOffsetY"+i,0);
					ij.Prefs.set("SMLocalizer.calibration.Biplane.ChOffsetZ"+i,0);
				}
				ij.Prefs.set("SMLocalizer.calibration.Biplane.window",5);
				ij.Prefs.set("SMLocalizer.calibration.Biplane.sigma",2);		
				ij.Prefs.set("SMLocalizer.calibration.Biplane.height",0);
				ij.Prefs.set("SMLocalizer.calibration.Biplane.channels",1);
				ij.Prefs.set("SMLocalizer.calibration.Biplane.step",0);
				ij.Prefs.set("SMLocalizer.calibration.Biplane.Ch1.0",0);
				for (int i = 1; i < 10; i++)
				{
					ij.Prefs.set("SMLocalizer.calibration.Astigmatism.ChOffsetX"+i,0);
					ij.Prefs.set("SMLocalizer.calibration.Astigmatism.ChOffsetY"+i,0);
					ij.Prefs.set("SMLocalizer.calibration.Astigmatism.ChOffsetZ"+i,0);
				}
				ij.Prefs.set("SMLocalizer.calibration.Astigmatism.window",7);
				ij.Prefs.set("SMLocalizer.calibration.Astigmatism.sigma",4);		
				ij.Prefs.set("SMLocalizer.calibration.Astigmatism.height",0);
				ij.Prefs.set("SMLocalizer.calibration.Astigmatism.channels",1);
				ij.Prefs.set("SMLocalizer.calibration.Astigmatism.step",0);
				ij.Prefs.set("SMLocalizer.calibration.Astigmatism.Ch1.0",0);
				ij.Prefs.set("SMLocalizer.calibration.Astigmatism.maxDim.Ch0",0);		
				ij.Prefs.savePreferences(); // store settings.
			 int id = 0; 		// set current ch to 1.
			 updateVisible(id);  // update fields that user can see.
			 String name = "default";
			 ij.Prefs.set("SMLocalizer.settingsEntries", 1);
			 ij.Prefs.set("SMLocalizer.settingsName"+1, name); // add storename
			 setParameters(name);


		 } finally{

		 }

	 }	
	                          
	 private void initComponents() {

	        pixelSizeChList = new javax.swing.JMenu();
	        pixelSizeData1 = new javax.swing.JMenuItem();
	        pixelSizeData2 = new javax.swing.JMenuItem();
	        pixelSizeData3 = new javax.swing.JMenuItem();
	        pixelSizeData4 = new javax.swing.JMenuItem();
	        pixelSizeData5 = new javax.swing.JMenuItem();
	        pixelSizeData6 = new javax.swing.JMenuItem();
	        pixelSizeData7 = new javax.swing.JMenuItem();
	        pixelSizeData8 = new javax.swing.JMenuItem();
	        pixelSizeData9 = new javax.swing.JMenuItem();
	        pixelSizeData10 = new javax.swing.JMenuItem();
	        totalGainChList = new javax.swing.JMenu();
	        totalGainData1 = new javax.swing.JMenuItem();
	        totalGainData2 = new javax.swing.JMenuItem();
	        totalGainData3 = new javax.swing.JMenuItem();
	        totalGainData4 = new javax.swing.JMenuItem();
	        totalGainData5 = new javax.swing.JMenuItem();
	        totalGainData6 = new javax.swing.JMenuItem();
	        totalGainData7 = new javax.swing.JMenuItem();
	        totalGainData8 = new javax.swing.JMenuItem();
	        totalGainData9 = new javax.swing.JMenuItem();
	        totalGainData10 = new javax.swing.JMenuItem();
	        minimalSignalChList = new javax.swing.JMenu();
	        minimalSignalData1 = new javax.swing.JMenuItem();
	        minimalSignalData2 = new javax.swing.JMenuItem();
	        minimalSignalData3 = new javax.swing.JMenuItem();
	        minimalSignalData4 = new javax.swing.JMenuItem();
	        minimalSignalData5 = new javax.swing.JMenuItem();
	        minimalSignalData6 = new javax.swing.JMenuItem();
	        minimalSignalData7 = new javax.swing.JMenuItem();
	        minimalSignalData8 = new javax.swing.JMenuItem();
	        minimalSignalData9 = new javax.swing.JMenuItem();
	        minimalSignalData10 = new javax.swing.JMenuItem();
	        gaussWindowChList = new javax.swing.JMenu();
	        ROIsizeData1 = new javax.swing.JMenuItem();
	        ROIsizeData2 = new javax.swing.JMenuItem();
	        ROIsizeData3 = new javax.swing.JMenuItem();
	        ROIsizeData4 = new javax.swing.JMenuItem();
	        ROIsizeData5 = new javax.swing.JMenuItem();
	        ROIsizeData6 = new javax.swing.JMenuItem();
	        ROIsizeData7 = new javax.swing.JMenuItem();
	        ROIsizeData8 = new javax.swing.JMenuItem();
	        ROIsizeData9 = new javax.swing.JMenuItem();
	        ROIsizeData10 = new javax.swing.JMenuItem();
	        windowWidthChList = new javax.swing.JMenu();
	        windowWidthData1 = new javax.swing.JMenuItem();
	        windowWidthData2 = new javax.swing.JMenuItem();
	        windowWidthData3 = new javax.swing.JMenuItem();
	        windowWidthData4 = new javax.swing.JMenuItem();
	        windowWidthData5 = new javax.swing.JMenuItem();
	        windowWidthData6 = new javax.swing.JMenuItem();
	        windowWidthData7 = new javax.swing.JMenuItem();
	        windowWidthData8 = new javax.swing.JMenuItem();
	        windowWidthData9 = new javax.swing.JMenuItem();
	        windowWidthData10 = new javax.swing.JMenuItem();
	        fiducialsChList = new javax.swing.JMenu();
	        fiducialsChoice1 = new javax.swing.JMenuItem();
	        fiducialsChoice2 = new javax.swing.JMenuItem();
	        fiducialsChoice3 = new javax.swing.JMenuItem();
	        fiducialsChoice4 = new javax.swing.JMenuItem();
	        fiducialsChoice5 = new javax.swing.JMenuItem();
	        fiducialsChoice6 = new javax.swing.JMenuItem();
	        fiducialsChoice7 = new javax.swing.JMenuItem();
	        fiducialsChoice8 = new javax.swing.JMenuItem();
	        fiducialsChoice9 = new javax.swing.JMenuItem();
	        fiducialsChoice10 = new javax.swing.JMenuItem();
	        doClusterAnalysisChList = new javax.swing.JMenu();
	        doClusterAnalysisData1 = new javax.swing.JMenuItem();
	        doClusterAnalysisData2 = new javax.swing.JMenuItem();
	        doClusterAnalysisData3 = new javax.swing.JMenuItem();
	        doClusterAnalysisData4 = new javax.swing.JMenuItem();
	        doClusterAnalysisData5 = new javax.swing.JMenuItem();
	        doClusterAnalysisData6 = new javax.swing.JMenuItem();
	        doClusterAnalysisData7 = new javax.swing.JMenuItem();
	        doClusterAnalysisData8 = new javax.swing.JMenuItem();
	        doClusterAnalysisData9 = new javax.swing.JMenuItem();
	        doClusterAnalysisData10 = new javax.swing.JMenuItem();
	        epsilonChList = new javax.swing.JMenu();
	        epsilonData1 = new javax.swing.JMenuItem();
	        epsilonData2 = new javax.swing.JMenuItem();
	        epsilonData3 = new javax.swing.JMenuItem();
	        epsilonData4 = new javax.swing.JMenuItem();
	        epsilonData5 = new javax.swing.JMenuItem();
	        epsilonData6 = new javax.swing.JMenuItem();
	        epsilonData7 = new javax.swing.JMenuItem();
	        epsilonData8 = new javax.swing.JMenuItem();
	        epsilonData9 = new javax.swing.JMenuItem();
	        epsilonData10 = new javax.swing.JMenuItem();
	        minPtsClusterChList = new javax.swing.JMenu();
	        minPtsClusterData1 = new javax.swing.JMenuItem();
	        minPtsClusterData2 = new javax.swing.JMenuItem();
	        minPtsClusterData3 = new javax.swing.JMenuItem();
	        minPtsClusterData4 = new javax.swing.JMenuItem();
	        minPtsClusterData5 = new javax.swing.JMenuItem();
	        minPtsClusterData6 = new javax.swing.JMenuItem();
	        minPtsClusterData7 = new javax.swing.JMenuItem();
	        minPtsClusterData8 = new javax.swing.JMenuItem();
	        minPtsClusterData9 = new javax.swing.JMenuItem();
	        minPtsClusterData10 = new javax.swing.JMenuItem();
	        outputPixelSizeChList = new javax.swing.JMenu();
	        outputPixelSizeData1 = new javax.swing.JMenuItem();
	        outputPixelSizeData2 = new javax.swing.JMenuItem();
	        outputPixelSizeData3 = new javax.swing.JMenuItem();
	        outputPixelSizeData4 = new javax.swing.JMenuItem();
	        outputPixelSizeData5 = new javax.swing.JMenuItem();
	        outputPixelSizeData6 = new javax.swing.JMenuItem();
	        outputPixelSizeData7 = new javax.swing.JMenuItem();
	        outputPixelSizeData8 = new javax.swing.JMenuItem();
	        outputPixelSizeData9 = new javax.swing.JMenuItem();
	        outputPixelSizeData10 = new javax.swing.JMenuItem();
	        driftCorrBinLowCountChList = new javax.swing.JMenu();
	        driftCorrBinLowCountData1 = new javax.swing.JMenuItem();
	        driftCorrBinLowCountData2 = new javax.swing.JMenuItem();
	        driftCorrBinLowCountData3 = new javax.swing.JMenuItem();
	        driftCorrBinLowCountData4 = new javax.swing.JMenuItem();
	        driftCorrBinLowCountData5 = new javax.swing.JMenuItem();
	        driftCorrBinLowCountData6 = new javax.swing.JMenuItem();
	        driftCorrBinLowCountData7 = new javax.swing.JMenuItem();
	        driftCorrBinLowCountData8 = new javax.swing.JMenuItem();
	        driftCorrBinLowCountData9 = new javax.swing.JMenuItem();
	        driftCorrBinLowCountData10 = new javax.swing.JMenuItem();
	        driftCorrBinHighCountChList = new javax.swing.JMenu();
	        driftCorrBinHighCountData1 = new javax.swing.JMenuItem();
	        driftCorrBinHighCountData2 = new javax.swing.JMenuItem();
	        driftCorrBinHighCountData3 = new javax.swing.JMenuItem();
	        driftCorrBinHighCountData4 = new javax.swing.JMenuItem();
	        driftCorrBinHighCountData5 = new javax.swing.JMenuItem();
	        driftCorrBinHighCountData6 = new javax.swing.JMenuItem();
	        driftCorrBinHighCountData7 = new javax.swing.JMenuItem();
	        driftCorrBinHighCountData8 = new javax.swing.JMenuItem();
	        driftCorrBinHighCountData9 = new javax.swing.JMenuItem();
	        driftCorrBinHighCountData10 = new javax.swing.JMenuItem();
	        numberOfBinsDriftCorrChList = new javax.swing.JMenu();
	        numberOfBinsDriftCorrData1 = new javax.swing.JMenuItem();
	        numberOfBinsDriftCorrData2 = new javax.swing.JMenuItem();
	        numberOfBinsDriftCorrData3 = new javax.swing.JMenuItem();
	        numberOfBinsDriftCorrData4 = new javax.swing.JMenuItem();
	        numberOfBinsDriftCorrData5 = new javax.swing.JMenuItem();
	        numberOfBinsDriftCorrData6 = new javax.swing.JMenuItem();
	        numberOfBinsDriftCorrData7 = new javax.swing.JMenuItem();
	        numberOfBinsDriftCorrData8 = new javax.swing.JMenuItem();
	        numberOfBinsDriftCorrData9 = new javax.swing.JMenuItem();
	        numberOfBinsDriftCorrData10 = new javax.swing.JMenuItem();
	        chAlignBinLowCountChList = new javax.swing.JMenu();
	        chAlignBinLowCountData1 = new javax.swing.JMenuItem();
	        chAlignBinLowCountData2 = new javax.swing.JMenuItem();
	        chAlignBinLowCountData3 = new javax.swing.JMenuItem();
	        chAlignBinLowCountData4 = new javax.swing.JMenuItem();
	        chAlignBinLowCountData5 = new javax.swing.JMenuItem();
	        chAlignBinLowCountData6 = new javax.swing.JMenuItem();
	        chAlignBinLowCountData7 = new javax.swing.JMenuItem();
	        chAlignBinLowCountData8 = new javax.swing.JMenuItem();
	        chAlignBinLowCountData9 = new javax.swing.JMenuItem();
	        chAlignBinLowCountData10 = new javax.swing.JMenuItem();
	        chAlignBinHighCountChList = new javax.swing.JMenu();
	        chAlignBinHighCountData1 = new javax.swing.JMenuItem();
	        chAlignBinHighCountData2 = new javax.swing.JMenuItem();
	        chAlignBinHighCountData3 = new javax.swing.JMenuItem();
	        chAlignBinHighCountData4 = new javax.swing.JMenuItem();
	        chAlignBinHighCountData5 = new javax.swing.JMenuItem();
	        chAlignBinHighCountData6 = new javax.swing.JMenuItem();
	        chAlignBinHighCountData7 = new javax.swing.JMenuItem();
	        chAlignBinHighCountData8 = new javax.swing.JMenuItem();
	        chAlignBinHighCountData9 = new javax.swing.JMenuItem();
	        chAlignBinHighCountData10 = new javax.swing.JMenuItem();
	        doPhotonCountChList = new javax.swing.JMenu();
	        doPhotonCountData1 = new javax.swing.JMenuItem();
	        doPhotonCountData2 = new javax.swing.JMenuItem();
	        doPhotonCountData3 = new javax.swing.JMenuItem();
	        doPhotonCountData4 = new javax.swing.JMenuItem();
	        doPhotonCountData5 = new javax.swing.JMenuItem();
	        doPhotonCountData6 = new javax.swing.JMenuItem();
	        doPhotonCountData7 = new javax.swing.JMenuItem();
	        doPhotonCountData8 = new javax.swing.JMenuItem();
	        doPhotonCountData9 = new javax.swing.JMenuItem();
	        doPhotonCountData10 = new javax.swing.JMenuItem();
	        minPhotonCountChList = new javax.swing.JMenu();
	        minPhotonCountData1 = new javax.swing.JMenuItem();
	        minPhotonCountData2 = new javax.swing.JMenuItem();
	        minPhotonCountData3 = new javax.swing.JMenuItem();
	        minPhotonCountData4 = new javax.swing.JMenuItem();
	        minPhotonCountData5 = new javax.swing.JMenuItem();
	        minPhotonCountData6 = new javax.swing.JMenuItem();
	        minPhotonCountData7 = new javax.swing.JMenuItem();
	        minPhotonCountData8 = new javax.swing.JMenuItem();
	        minPhotonCountData9 = new javax.swing.JMenuItem();
	        minPhotonCountData10 = new javax.swing.JMenuItem();
	        maxPhotonCountChList = new javax.swing.JMenu();
	        maxPhotonCountData1 = new javax.swing.JMenuItem();
	        maxPhotonCountData2 = new javax.swing.JMenuItem();
	        maxPhotonCountData3 = new javax.swing.JMenuItem();
	        maxPhotonCountData4 = new javax.swing.JMenuItem();
	        maxPhotonCountData5 = new javax.swing.JMenuItem();
	        maxPhotonCountData6 = new javax.swing.JMenuItem();
	        maxPhotonCountData7 = new javax.swing.JMenuItem();
	        maxPhotonCountData8 = new javax.swing.JMenuItem();
	        maxPhotonCountData9 = new javax.swing.JMenuItem();
	        maxPhotonCountData10 = new javax.swing.JMenuItem();
	        doSigmaXYChList = new javax.swing.JMenu();
	        doSigmaXYData1 = new javax.swing.JMenuItem();
	        doSigmaXYData2 = new javax.swing.JMenuItem();
	        doSigmaXYData3 = new javax.swing.JMenuItem();
	        doSigmaXYData4 = new javax.swing.JMenuItem();
	        doSigmaXYData5 = new javax.swing.JMenuItem();
	        doSigmaXYData6 = new javax.swing.JMenuItem();
	        doSigmaXYData7 = new javax.swing.JMenuItem();
	        doSigmaXYData8 = new javax.swing.JMenuItem();
	        doSigmaXYData9 = new javax.swing.JMenuItem();
	        doSigmaXYData10 = new javax.swing.JMenuItem();
	        minSigmaXYChList = new javax.swing.JMenu();
	        minSigmaXYData1 = new javax.swing.JMenuItem();
	        minSigmaXYData2 = new javax.swing.JMenuItem();
	        minSigmaXYData3 = new javax.swing.JMenuItem();
	        minSigmaXYData4 = new javax.swing.JMenuItem();
	        minSigmaXYData5 = new javax.swing.JMenuItem();
	        minSigmaXYData6 = new javax.swing.JMenuItem();
	        minSigmaXYData7 = new javax.swing.JMenuItem();
	        minSigmaXYData8 = new javax.swing.JMenuItem();
	        minSigmaXYData9 = new javax.swing.JMenuItem();
	        minSigmaXYData10 = new javax.swing.JMenuItem();
	        maxSigmaXYChList = new javax.swing.JMenu();
	        maxSigmaXYData1 = new javax.swing.JMenuItem();
	        maxSigmaXYData2 = new javax.swing.JMenuItem();
	        maxSigmaXYData3 = new javax.swing.JMenuItem();
	        maxSigmaXYData4 = new javax.swing.JMenuItem();
	        maxSigmaXYData5 = new javax.swing.JMenuItem();
	        maxSigmaXYData6 = new javax.swing.JMenuItem();
	        maxSigmaXYData7 = new javax.swing.JMenuItem();
	        maxSigmaXYData8 = new javax.swing.JMenuItem();
	        maxSigmaXYData9 = new javax.swing.JMenuItem();
	        maxSigmaXYData10 = new javax.swing.JMenuItem();
	        doSigmaZChList = new javax.swing.JMenu();
	        doSigmaZData1 = new javax.swing.JMenuItem();
	        doSigmaZData2 = new javax.swing.JMenuItem();
	        doSigmaZData3 = new javax.swing.JMenuItem();
	        doSigmaZData4 = new javax.swing.JMenuItem();
	        doSigmaZData5 = new javax.swing.JMenuItem();
	        doSigmaZData6 = new javax.swing.JMenuItem();
	        doSigmaZData7 = new javax.swing.JMenuItem();
	        doSigmaZData8 = new javax.swing.JMenuItem();
	        doSigmaZData9 = new javax.swing.JMenuItem();
	        doSigmaZData10 = new javax.swing.JMenuItem();
	        minSigmaZChList = new javax.swing.JMenu();
	        minSigmaZData1 = new javax.swing.JMenuItem();
	        minSigmaZData2 = new javax.swing.JMenuItem();
	        minSigmaZData3 = new javax.swing.JMenuItem();
	        minSigmaZData4 = new javax.swing.JMenuItem();
	        minSigmaZData5 = new javax.swing.JMenuItem();
	        minSigmaZData6 = new javax.swing.JMenuItem();
	        minSigmaZData7 = new javax.swing.JMenuItem();
	        minSigmaZData8 = new javax.swing.JMenuItem();
	        minSigmaZData9 = new javax.swing.JMenuItem();
	        minSigmaZData10 = new javax.swing.JMenuItem();
	        maxSigmaZChList = new javax.swing.JMenu();
	        maxSigmaZData1 = new javax.swing.JMenuItem();
	        maxSigmaZData2 = new javax.swing.JMenuItem();
	        maxSigmaZData3 = new javax.swing.JMenuItem();
	        maxSigmaZData4 = new javax.swing.JMenuItem();
	        maxSigmaZData5 = new javax.swing.JMenuItem();
	        maxSigmaZData6 = new javax.swing.JMenuItem();
	        maxSigmaZData7 = new javax.swing.JMenuItem();
	        maxSigmaZData8 = new javax.swing.JMenuItem();
	        maxSigmaZData9 = new javax.swing.JMenuItem();
	        maxSigmaZData10 = new javax.swing.JMenuItem();
	        doRsquareChList = new javax.swing.JMenu();
	        doRsquareData1 = new javax.swing.JMenuItem();
	        doRsquareData2 = new javax.swing.JMenuItem();
	        doRsquareData3 = new javax.swing.JMenuItem();
	        doRsquareData4 = new javax.swing.JMenuItem();
	        doRsquareData5 = new javax.swing.JMenuItem();
	        doRsquareData6 = new javax.swing.JMenuItem();
	        doRsquareData7 = new javax.swing.JMenuItem();
	        doRsquareData8 = new javax.swing.JMenuItem();
	        doRsquareData9 = new javax.swing.JMenuItem();
	        doRsquareData10 = new javax.swing.JMenuItem();
	        minRsquareChList = new javax.swing.JMenu();
	        minRsquareData1 = new javax.swing.JMenuItem();
	        minRsquareData2 = new javax.swing.JMenuItem();
	        minRsquareData3 = new javax.swing.JMenuItem();
	        minRsquareData4 = new javax.swing.JMenuItem();
	        minRsquareData5 = new javax.swing.JMenuItem();
	        minRsquareData6 = new javax.swing.JMenuItem();
	        minRsquareData7 = new javax.swing.JMenuItem();
	        minRsquareData8 = new javax.swing.JMenuItem();
	        minRsquareData9 = new javax.swing.JMenuItem();
	        minRsquareData10 = new javax.swing.JMenuItem();
	        maxRsquareChList = new javax.swing.JMenu();
	        maxRsquareData1 = new javax.swing.JMenuItem();
	        maxRsquareData2 = new javax.swing.JMenuItem();
	        maxRsquareData3 = new javax.swing.JMenuItem();
	        maxRsquareData4 = new javax.swing.JMenuItem();
	        maxRsquareData5 = new javax.swing.JMenuItem();
	        maxRsquareData6 = new javax.swing.JMenuItem();
	        maxRsquareData7 = new javax.swing.JMenuItem();
	        maxRsquareData8 = new javax.swing.JMenuItem();
	        maxRsquareData9 = new javax.swing.JMenuItem();
	        maxRsquareData10 = new javax.swing.JMenuItem();
	        doPrecisionXYChList = new javax.swing.JMenu();
	        doPrecisionXYData1 = new javax.swing.JMenuItem();
	        doPrecisionXYData2 = new javax.swing.JMenuItem();
	        doPrecisionXYData3 = new javax.swing.JMenuItem();
	        doPrecisionXYData4 = new javax.swing.JMenuItem();
	        doPrecisionXYData5 = new javax.swing.JMenuItem();
	        doPrecisionXYData6 = new javax.swing.JMenuItem();
	        doPrecisionXYData7 = new javax.swing.JMenuItem();
	        doPrecisionXYData8 = new javax.swing.JMenuItem();
	        doPrecisionXYData9 = new javax.swing.JMenuItem();
	        doPrecisionXYData10 = new javax.swing.JMenuItem();
	        minPrecisionXYChList = new javax.swing.JMenu();
	        minPrecisionXYData1 = new javax.swing.JMenuItem();
	        minPrecisionXYData2 = new javax.swing.JMenuItem();
	        minPrecisionXYData3 = new javax.swing.JMenuItem();
	        minPrecisionXYData4 = new javax.swing.JMenuItem();
	        minPrecisionXYData5 = new javax.swing.JMenuItem();
	        minPrecisionXYData6 = new javax.swing.JMenuItem();
	        minPrecisionXYData7 = new javax.swing.JMenuItem();
	        minPrecisionXYData8 = new javax.swing.JMenuItem();
	        minPrecisionXYData9 = new javax.swing.JMenuItem();
	        minPrecisionXYData10 = new javax.swing.JMenuItem();
	        maxPrecisionXYChList = new javax.swing.JMenu();
	        maxPrecisionXYData1 = new javax.swing.JMenuItem();
	        maxPrecisionXYData2 = new javax.swing.JMenuItem();
	        maxPrecisionXYData3 = new javax.swing.JMenuItem();
	        maxPrecisionXYData4 = new javax.swing.JMenuItem();
	        maxPrecisionXYData5 = new javax.swing.JMenuItem();
	        maxPrecisionXYData6 = new javax.swing.JMenuItem();
	        maxPrecisionXYData7 = new javax.swing.JMenuItem();
	        maxPrecisionXYData8 = new javax.swing.JMenuItem();
	        maxPrecisionXYData9 = new javax.swing.JMenuItem();
	        maxPrecisionXYData10 = new javax.swing.JMenuItem();
	        doPrecisionZChList = new javax.swing.JMenu();
	        doPrecisionZData1 = new javax.swing.JMenuItem();
	        doPrecisionZData2 = new javax.swing.JMenuItem();
	        doPrecisionZData3 = new javax.swing.JMenuItem();
	        doPrecisionZData4 = new javax.swing.JMenuItem();
	        doPrecisionZData5 = new javax.swing.JMenuItem();
	        doPrecisionZData6 = new javax.swing.JMenuItem();
	        doPrecisionZData7 = new javax.swing.JMenuItem();
	        doPrecisionZData8 = new javax.swing.JMenuItem();
	        doPrecisionZData9 = new javax.swing.JMenuItem();
	        doPrecisionZData10 = new javax.swing.JMenuItem();
	        minPrecisionZChList = new javax.swing.JMenu();
	        minPrecisionZData1 = new javax.swing.JMenuItem();
	        minPrecisionZData2 = new javax.swing.JMenuItem();
	        minPrecisionZData3 = new javax.swing.JMenuItem();
	        minPrecisionZData4 = new javax.swing.JMenuItem();
	        minPrecisionZData5 = new javax.swing.JMenuItem();
	        minPrecisionZData6 = new javax.swing.JMenuItem();
	        minPrecisionZData7 = new javax.swing.JMenuItem();
	        minPrecisionZData8 = new javax.swing.JMenuItem();
	        minPrecisionZData9 = new javax.swing.JMenuItem();
	        minPrecisionZData10 = new javax.swing.JMenuItem();
	        maxPrecisionZChList = new javax.swing.JMenu();
	        maxPrecisionZData1 = new javax.swing.JMenuItem();
	        maxPrecisionZData2 = new javax.swing.JMenuItem();
	        maxPrecisionZData3 = new javax.swing.JMenuItem();
	        maxPrecisionZData4 = new javax.swing.JMenuItem();
	        maxPrecisionZData5 = new javax.swing.JMenuItem();
	        maxPrecisionZData6 = new javax.swing.JMenuItem();
	        maxPrecisionZData7 = new javax.swing.JMenuItem();
	        maxPrecisionZData8 = new javax.swing.JMenuItem();
	        maxPrecisionZData9 = new javax.swing.JMenuItem();
	        maxPrecisionZData10 = new javax.swing.JMenuItem();
	        driftCorrShiftXYChList = new javax.swing.JMenu();
	        driftCorrShiftXYData1 = new javax.swing.JMenuItem();
	        driftCorrShiftXYData2 = new javax.swing.JMenuItem();
	        driftCorrShiftXYData3 = new javax.swing.JMenuItem();
	        driftCorrShiftXYData4 = new javax.swing.JMenuItem();
	        driftCorrShiftXYData5 = new javax.swing.JMenuItem();
	        driftCorrShiftXYData6 = new javax.swing.JMenuItem();
	        driftCorrShiftXYData7 = new javax.swing.JMenuItem();
	        driftCorrShiftXYData8 = new javax.swing.JMenuItem();
	        driftCorrShiftXYData9 = new javax.swing.JMenuItem();
	        driftCorrShiftXYData10 = new javax.swing.JMenuItem();
	        driftCorrShiftZChList = new javax.swing.JMenu();
	        driftCorrShiftZData1 = new javax.swing.JMenuItem();
	        driftCorrShiftZData2 = new javax.swing.JMenuItem();
	        driftCorrShiftZData3 = new javax.swing.JMenuItem();
	        driftCorrShiftZData4 = new javax.swing.JMenuItem();
	        driftCorrShiftZData5 = new javax.swing.JMenuItem();
	        driftCorrShiftZData6 = new javax.swing.JMenuItem();
	        driftCorrShiftZData7 = new javax.swing.JMenuItem();
	        driftCorrShiftZData8 = new javax.swing.JMenuItem();
	        driftCorrShiftZData9 = new javax.swing.JMenuItem();
	        driftCorrShiftZData10 = new javax.swing.JMenuItem();
	        chAlignShiftXYChList = new javax.swing.JMenu();
	        chAlignShiftXYData1 = new javax.swing.JMenuItem();
	        chAlignShiftXYData2 = new javax.swing.JMenuItem();
	        chAlignShiftXYData3 = new javax.swing.JMenuItem();
	        chAlignShiftXYData4 = new javax.swing.JMenuItem();
	        chAlignShiftXYData5 = new javax.swing.JMenuItem();
	        chAlignShiftXYData6 = new javax.swing.JMenuItem();
	        chAlignShiftXYData7 = new javax.swing.JMenuItem();
	        chAlignShiftXYData8 = new javax.swing.JMenuItem();
	        chAlignShiftXYData9 = new javax.swing.JMenuItem();
	        chAlignShiftXYData10 = new javax.swing.JMenuItem();
	        chAlignShiftZChList = new javax.swing.JMenu();
	        chAlignShiftZData1 = new javax.swing.JMenuItem();
	        chAlignShiftZData2 = new javax.swing.JMenuItem();
	        chAlignShiftZData3 = new javax.swing.JMenuItem();
	        chAlignShiftZData4 = new javax.swing.JMenuItem();
	        chAlignShiftZData5 = new javax.swing.JMenuItem();
	        chAlignShiftZData6 = new javax.swing.JMenuItem();
	        chAlignShiftZData7 = new javax.swing.JMenuItem();
	        chAlignShiftZData8 = new javax.swing.JMenuItem();
	        chAlignShiftZData9 = new javax.swing.JMenuItem();
	        chAlignShiftZData10 = new javax.swing.JMenuItem();
	        buttonGroup2 = new javax.swing.ButtonGroup();
	        doFrameChList = new javax.swing.JMenu();
	        doFrameData1 = new javax.swing.JMenuItem();
	        doFrameData2 = new javax.swing.JMenuItem();
	        doFrameData3 = new javax.swing.JMenuItem();
	        doFrameData4 = new javax.swing.JMenuItem();
	        doFrameData5 = new javax.swing.JMenuItem();
	        doFrameData6 = new javax.swing.JMenuItem();
	        doFrameData7 = new javax.swing.JMenuItem();
	        doFrameData8 = new javax.swing.JMenuItem();
	        doFrameData9 = new javax.swing.JMenuItem();
	        doFrameData10 = new javax.swing.JMenuItem();
	        minFrameChList = new javax.swing.JMenu();
	        minFrameData1 = new javax.swing.JMenuItem();
	        minFrameData2 = new javax.swing.JMenuItem();
	        minFrameData3 = new javax.swing.JMenuItem();
	        minFrameData4 = new javax.swing.JMenuItem();
	        minFrameData5 = new javax.swing.JMenuItem();
	        minFrameData6 = new javax.swing.JMenuItem();
	        minFrameData7 = new javax.swing.JMenuItem();
	        minFrameData8 = new javax.swing.JMenuItem();
	        minFrameData9 = new javax.swing.JMenuItem();
	        minFrameData10 = new javax.swing.JMenuItem();
	        maxFrameChList = new javax.swing.JMenu();
	        maxFrameData1 = new javax.swing.JMenuItem();
	        maxFrameData2 = new javax.swing.JMenuItem();
	        maxFrameData3 = new javax.swing.JMenuItem();
	        maxFrameData4 = new javax.swing.JMenuItem();
	        maxFrameData5 = new javax.swing.JMenuItem();
	        maxFrameData6 = new javax.swing.JMenuItem();
	        maxFrameData7 = new javax.swing.JMenuItem();
	        maxFrameData8 = new javax.swing.JMenuItem();
	        maxFrameData9 = new javax.swing.JMenuItem();
	        maxFrameData10 = new javax.swing.JMenuItem();
	        doRenderImageChList = new javax.swing.JMenu();
	        doRenderImageData1 = new javax.swing.JMenuItem();
	        doRenderImageData2 = new javax.swing.JMenuItem();
	        doRenderImageData3 = new javax.swing.JMenuItem();
	        doRenderImageData4 = new javax.swing.JMenuItem();
	        doRenderImageData5 = new javax.swing.JMenuItem();
	        doRenderImageData6 = new javax.swing.JMenuItem();
	        doRenderImageData7 = new javax.swing.JMenuItem();
	        doRenderImageData8 = new javax.swing.JMenuItem();
	        doRenderImageData9 = new javax.swing.JMenuItem();
	        doRenderImageData10 = new javax.swing.JMenuItem();
	        doCorrelativeChList = new javax.swing.JMenu();
	        doCorrelativeData1 = new javax.swing.JMenuItem();
	        doCorrelativeData2 = new javax.swing.JMenuItem();
	        doCorrelativeData3 = new javax.swing.JMenuItem();
	        doCorrelativeData4 = new javax.swing.JMenuItem();
	        doCorrelativeData5 = new javax.swing.JMenuItem();
	        doCorrelativeData6 = new javax.swing.JMenuItem();
	        doCorrelativeData7 = new javax.swing.JMenuItem();
	        doCorrelativeData8 = new javax.swing.JMenuItem();
	        doCorrelativeData9 = new javax.swing.JMenuItem();
	        doCorrelativeData10 = new javax.swing.JMenuItem();
	        doChromaticChList = new javax.swing.JMenu();
	        doChromaticData1 = new javax.swing.JMenuItem();
	        doChromaticData2 = new javax.swing.JMenuItem();
	        doChromaticData3 = new javax.swing.JMenuItem();
	        doChromaticData4 = new javax.swing.JMenuItem();
	        doChromaticData5 = new javax.swing.JMenuItem();
	        doChromaticData6 = new javax.swing.JMenuItem();
	        doChromaticData7 = new javax.swing.JMenuItem();
	        doChromaticData8 = new javax.swing.JMenuItem();
	        doChromaticData9 = new javax.swing.JMenuItem();
	        doChromaticData10 = new javax.swing.JMenuItem();
	        Header = new javax.swing.JLabel();
	        BasicInp = new javax.swing.JPanel();
	        basicInput = new javax.swing.JLabel();
	        inputPixelSizeLabel = new javax.swing.JLabel();
	        inputPixelSize = new javax.swing.JTextField();
	        totalGainLabel = new javax.swing.JLabel();
	        totalGain = new javax.swing.JTextField();
	        minimalSignalLabel = new javax.swing.JLabel();
	        minimalSignal = new javax.swing.JTextField();
	        windowWidthLabel = new javax.swing.JLabel();
	        windowWidth = new javax.swing.JTextField();
	        Process = new javax.swing.JButton();
	        resetBasicInput = new javax.swing.JButton();
	        calibrate = new javax.swing.JButton();
	        modality = new javax.swing.JComboBox<>();
	        channelId = new javax.swing.JComboBox<>();
	        ParameterRange = new javax.swing.JPanel();
	        ParameterLabel = new javax.swing.JLabel();
	        doPhotonCount = new javax.swing.JCheckBox();
	        doSigmaXY = new javax.swing.JCheckBox();
	        doRsquare = new javax.swing.JCheckBox();
	        doPrecisionXY = new javax.swing.JCheckBox();
	        doPrecisionZ = new javax.swing.JCheckBox();
	        cleanTable = new javax.swing.JButton();
	        minPrecisionXY = new javax.swing.JTextField();
	        maxPrecisionXY = new javax.swing.JTextField();
	        minRsquare = new javax.swing.JTextField();
	        maxRsquare = new javax.swing.JTextField();
	        minSigmaXY = new javax.swing.JTextField();
	        maxSigmaXY = new javax.swing.JTextField();
	        minPhotonCount = new javax.swing.JTextField();
	        maxPhotonCount = new javax.swing.JTextField();
	        minPrecisionZ = new javax.swing.JTextField();
	        maxPrecisionZ = new javax.swing.JTextField();
	        resetParameterRange = new javax.swing.JButton();
	        maxLabel3 = new javax.swing.JLabel();
	        minLabel3 = new javax.swing.JLabel();
	        doFrame = new javax.swing.JCheckBox();
	        minFrame = new javax.swing.JTextField();
	        maxFrame = new javax.swing.JTextField();
	        Analysis = new javax.swing.JPanel();
	        doClusterAnalysis = new javax.swing.JCheckBox();
	        epsilonLabel = new javax.swing.JLabel();
	        minPtsLabel = new javax.swing.JLabel();
	        epsilon = new javax.swing.JTextField();
	        minPtsCluster = new javax.swing.JTextField();
	        clusterAnalysis = new javax.swing.JButton();
	        jPanel1 = new javax.swing.JPanel();
	        particlesPerBinLabel = new javax.swing.JLabel();
	        driftCorrBinLowCount = new javax.swing.JTextField();
	        driftCorrBinHighCount = new javax.swing.JTextField();
	        numberOfBinsLabel = new javax.swing.JLabel();
	        doDriftCorrect = new javax.swing.JCheckBox();
	        driftCorrect = new javax.swing.JButton();
	        numberOfBinsDriftCorr = new javax.swing.JTextField();
	        minLabel = new javax.swing.JLabel();
	        maxLabel = new javax.swing.JLabel();
	        particlesPerBinLabel1 = new javax.swing.JLabel();
	        driftCorrShiftXY = new javax.swing.JTextField();
	        driftCorrShiftZ = new javax.swing.JTextField();
	        minLabel2 = new javax.swing.JLabel();
	        maxLabel2 = new javax.swing.JLabel();
	        jPanel2 = new javax.swing.JPanel();
	        correctBackground = new javax.swing.JButton();
	        localize_Fit = new javax.swing.JButton();
	        loadSettings = new javax.swing.JButton();
	        storeSettings = new javax.swing.JButton();
	        jPanel3 = new javax.swing.JPanel();
	        renderImage = new javax.swing.JButton();
	        doRenderImage = new javax.swing.JCheckBox();
	        outputPixelSizeLabel = new javax.swing.JLabel();
	        outputPixelSize = new javax.swing.JTextField();
	        doGaussianSmoothing = new javax.swing.JCheckBox();
	        outputPixelSizeZ = new javax.swing.JTextField();
	        XYrenderLabel = new javax.swing.JLabel();
	        ZrenderLabel = new javax.swing.JLabel();
	        parallelComputation = new javax.swing.JRadioButton();
	        GPUcomputation = new javax.swing.JRadioButton();
	        jPanel4 = new javax.swing.JPanel();
	        doChannelAlign = new javax.swing.JCheckBox();
	        alignChannels = new javax.swing.JButton();
	        particlesPerBinLabelchAlign = new javax.swing.JLabel();
	        chAlignBinLowCount = new javax.swing.JTextField();
	        minLabel1 = new javax.swing.JLabel();
	        maxLabel1 = new javax.swing.JLabel();
	        chAlignBinHighCount = new javax.swing.JTextField();
	        chAlignShiftZ = new javax.swing.JTextField();
	        maxLabel4 = new javax.swing.JLabel();
	        chAlignShiftXY = new javax.swing.JTextField();
	        minLabel4 = new javax.swing.JLabel();
	        particlesPerBinLabel2 = new javax.swing.JLabel();

	        pixelSizeChList.setText("jMenu1");

	        pixelSizeData1.setText("100");
	        pixelSizeData1.setToolTipText("");
	        pixelSizeChList.add(pixelSizeData1);

	        pixelSizeData2.setText("100");
	        pixelSizeData2.setToolTipText("");
	        pixelSizeChList.add(pixelSizeData2);

	        pixelSizeData3.setText("100");
	        pixelSizeData3.setToolTipText("");
	        pixelSizeChList.add(pixelSizeData3);

	        pixelSizeData4.setText("100");
	        pixelSizeData4.setToolTipText("");
	        pixelSizeChList.add(pixelSizeData4);

	        pixelSizeData5.setText("100");
	        pixelSizeData5.setToolTipText("");
	        pixelSizeChList.add(pixelSizeData5);

	        pixelSizeData6.setText("100");
	        pixelSizeData6.setToolTipText("");
	        pixelSizeChList.add(pixelSizeData6);

	        pixelSizeData7.setText("100");
	        pixelSizeData7.setToolTipText("");
	        pixelSizeChList.add(pixelSizeData7);

	        pixelSizeData8.setText("100");
	        pixelSizeData8.setToolTipText("");
	        pixelSizeChList.add(pixelSizeData8);

	        pixelSizeData9.setText("100");
	        pixelSizeData9.setToolTipText("");
	        pixelSizeChList.add(pixelSizeData9);

	        pixelSizeData10.setText("100");
	        pixelSizeData10.setToolTipText("");
	        pixelSizeChList.add(pixelSizeData10);

	        totalGainChList.setText("jMenu1");

	        totalGainData1.setText("100");
	        totalGainData1.setToolTipText("");
	        totalGainChList.add(totalGainData1);

	        totalGainData2.setText("100");
	        totalGainData2.setToolTipText("");
	        totalGainChList.add(totalGainData2);

	        totalGainData3.setText("100");
	        totalGainData3.setToolTipText("");
	        totalGainChList.add(totalGainData3);

	        totalGainData4.setText("100");
	        totalGainData4.setToolTipText("");
	        totalGainChList.add(totalGainData4);

	        totalGainData5.setText("100");
	        totalGainData5.setToolTipText("");
	        totalGainChList.add(totalGainData5);

	        totalGainData6.setText("100");
	        totalGainData6.setToolTipText("");
	        totalGainChList.add(totalGainData6);

	        totalGainData7.setText("100");
	        totalGainData7.setToolTipText("");
	        totalGainChList.add(totalGainData7);

	        totalGainData8.setText("100");
	        totalGainData8.setToolTipText("");
	        totalGainChList.add(totalGainData8);

	        totalGainData9.setText("100");
	        totalGainData9.setToolTipText("");
	        totalGainChList.add(totalGainData9);

	        totalGainData10.setText("100");
	        totalGainData10.setToolTipText("");
	        totalGainChList.add(totalGainData10);

	        minimalSignalChList.setText("jMenu1");

	        minimalSignalData1.setText("100");
	        minimalSignalData1.setToolTipText("");
	        minimalSignalChList.add(minimalSignalData1);

	        minimalSignalData2.setText("100");
	        minimalSignalData2.setToolTipText("");
	        minimalSignalChList.add(minimalSignalData2);

	        minimalSignalData3.setText("100");
	        minimalSignalData3.setToolTipText("");
	        minimalSignalChList.add(minimalSignalData3);

	        minimalSignalData4.setText("100");
	        minimalSignalData4.setToolTipText("");
	        minimalSignalChList.add(minimalSignalData4);

	        minimalSignalData5.setText("100");
	        minimalSignalData5.setToolTipText("");
	        minimalSignalChList.add(minimalSignalData5);

	        minimalSignalData6.setText("100");
	        minimalSignalData6.setToolTipText("");
	        minimalSignalChList.add(minimalSignalData6);

	        minimalSignalData7.setText("100");
	        minimalSignalData7.setToolTipText("");
	        minimalSignalChList.add(minimalSignalData7);

	        minimalSignalData8.setText("100");
	        minimalSignalData8.setToolTipText("");
	        minimalSignalChList.add(minimalSignalData8);

	        minimalSignalData9.setText("100");
	        minimalSignalData9.setToolTipText("");
	        minimalSignalChList.add(minimalSignalData9);

	        minimalSignalData10.setText("100");
	        minimalSignalData10.setToolTipText("");
	        minimalSignalChList.add(minimalSignalData10);

	        gaussWindowChList.setText("jMenu1");

	        ROIsizeData1.setText("100");
	        ROIsizeData1.setToolTipText("");
	        gaussWindowChList.add(ROIsizeData1);

	        ROIsizeData2.setText("100");
	        ROIsizeData2.setToolTipText("");
	        gaussWindowChList.add(ROIsizeData2);

	        ROIsizeData3.setText("100");
	        ROIsizeData3.setToolTipText("");
	        gaussWindowChList.add(ROIsizeData3);

	        ROIsizeData4.setText("100");
	        ROIsizeData4.setToolTipText("");
	        gaussWindowChList.add(ROIsizeData4);

	        ROIsizeData5.setText("100");
	        ROIsizeData5.setToolTipText("");
	        gaussWindowChList.add(ROIsizeData5);

	        ROIsizeData6.setText("100");
	        ROIsizeData6.setToolTipText("");
	        gaussWindowChList.add(ROIsizeData6);

	        ROIsizeData7.setText("100");
	        ROIsizeData7.setToolTipText("");
	        gaussWindowChList.add(ROIsizeData7);

	        ROIsizeData8.setText("100");
	        ROIsizeData8.setToolTipText("");
	        gaussWindowChList.add(ROIsizeData8);

	        ROIsizeData9.setText("100");
	        ROIsizeData9.setToolTipText("");
	        gaussWindowChList.add(ROIsizeData9);

	        ROIsizeData10.setText("100");
	        ROIsizeData10.setToolTipText("");
	        gaussWindowChList.add(ROIsizeData10);

	        windowWidthChList.setText("jMenu1");

	        windowWidthData1.setText("jMenuItem1");
	        windowWidthChList.add(windowWidthData1);

	        windowWidthData2.setText("jMenuItem1");
	        windowWidthChList.add(windowWidthData2);

	        windowWidthData3.setText("jMenuItem1");
	        windowWidthChList.add(windowWidthData3);

	        windowWidthData4.setText("jMenuItem1");
	        windowWidthChList.add(windowWidthData4);

	        windowWidthData5.setText("jMenuItem1");
	        windowWidthChList.add(windowWidthData5);

	        windowWidthData6.setText("jMenuItem1");
	        windowWidthChList.add(windowWidthData6);

	        windowWidthData7.setText("jMenuItem1");
	        windowWidthChList.add(windowWidthData7);

	        windowWidthData8.setText("jMenuItem1");
	        windowWidthChList.add(windowWidthData8);

	        windowWidthData9.setText("jMenuItem1");
	        windowWidthChList.add(windowWidthData9);

	        windowWidthData10.setText("jMenuItem1");
	        windowWidthChList.add(windowWidthData10);

	        fiducialsChList.setText("jMenu1");

	        fiducialsChoice1.setText("jMenuItem1");
	        fiducialsChList.add(fiducialsChoice1);

	        fiducialsChoice2.setText("jMenuItem1");
	        fiducialsChList.add(fiducialsChoice2);

	        fiducialsChoice3.setText("jMenuItem1");
	        fiducialsChList.add(fiducialsChoice3);

	        fiducialsChoice4.setText("jMenuItem1");
	        fiducialsChList.add(fiducialsChoice4);

	        fiducialsChoice5.setText("jMenuItem1");
	        fiducialsChList.add(fiducialsChoice5);

	        fiducialsChoice6.setText("jMenuItem1");
	        fiducialsChList.add(fiducialsChoice6);

	        fiducialsChoice7.setText("jMenuItem1");
	        fiducialsChList.add(fiducialsChoice7);

	        fiducialsChoice8.setText("jMenuItem1");
	        fiducialsChList.add(fiducialsChoice8);

	        fiducialsChoice9.setText("jMenuItem1");
	        fiducialsChList.add(fiducialsChoice9);

	        fiducialsChoice10.setText("jMenuItem1");
	        fiducialsChList.add(fiducialsChoice10);

	        doClusterAnalysisChList.setText("jMenu1");

	        doClusterAnalysisData1.setText("jMenuItem1");
	        doClusterAnalysisChList.add(doClusterAnalysisData1);

	        doClusterAnalysisData2.setText("jMenuItem1");
	        doClusterAnalysisChList.add(doClusterAnalysisData2);

	        doClusterAnalysisData3.setText("jMenuItem1");
	        doClusterAnalysisChList.add(doClusterAnalysisData3);

	        doClusterAnalysisData4.setText("jMenuItem1");
	        doClusterAnalysisChList.add(doClusterAnalysisData4);

	        doClusterAnalysisData5.setText("jMenuItem1");
	        doClusterAnalysisChList.add(doClusterAnalysisData5);

	        doClusterAnalysisData6.setText("jMenuItem1");
	        doClusterAnalysisChList.add(doClusterAnalysisData6);

	        doClusterAnalysisData7.setText("jMenuItem1");
	        doClusterAnalysisChList.add(doClusterAnalysisData7);

	        doClusterAnalysisData8.setText("jMenuItem1");
	        doClusterAnalysisChList.add(doClusterAnalysisData8);

	        doClusterAnalysisData9.setText("jMenuItem1");
	        doClusterAnalysisChList.add(doClusterAnalysisData9);

	        doClusterAnalysisData10.setText("jMenuItem1");
	        doClusterAnalysisChList.add(doClusterAnalysisData10);

	        epsilonChList.setText("jMenu1");

	        epsilonData1.setText("jMenuItem1");
	        epsilonChList.add(epsilonData1);

	        epsilonData2.setText("jMenuItem1");
	        epsilonChList.add(epsilonData2);

	        epsilonData3.setText("jMenuItem1");
	        epsilonChList.add(epsilonData3);

	        epsilonData4.setText("jMenuItem1");
	        epsilonChList.add(epsilonData4);

	        epsilonData5.setText("jMenuItem1");
	        epsilonChList.add(epsilonData5);

	        epsilonData6.setText("jMenuItem1");
	        epsilonChList.add(epsilonData6);

	        epsilonData7.setText("jMenuItem1");
	        epsilonChList.add(epsilonData7);

	        epsilonData8.setText("jMenuItem1");
	        epsilonChList.add(epsilonData8);

	        epsilonData9.setText("jMenuItem1");
	        epsilonChList.add(epsilonData9);

	        epsilonData10.setText("jMenuItem1");
	        epsilonChList.add(epsilonData10);

	        minPtsClusterChList.setText("jMenu1");

	        minPtsClusterData1.setText("jMenuItem1");
	        minPtsClusterChList.add(minPtsClusterData1);

	        minPtsClusterData2.setText("jMenuItem1");
	        minPtsClusterChList.add(minPtsClusterData2);

	        minPtsClusterData3.setText("jMenuItem1");
	        minPtsClusterChList.add(minPtsClusterData3);

	        minPtsClusterData4.setText("jMenuItem1");
	        minPtsClusterChList.add(minPtsClusterData4);

	        minPtsClusterData5.setText("jMenuItem1");
	        minPtsClusterChList.add(minPtsClusterData5);

	        minPtsClusterData6.setText("jMenuItem1");
	        minPtsClusterChList.add(minPtsClusterData6);

	        minPtsClusterData7.setText("jMenuItem1");
	        minPtsClusterChList.add(minPtsClusterData7);

	        minPtsClusterData8.setText("jMenuItem1");
	        minPtsClusterChList.add(minPtsClusterData8);

	        minPtsClusterData9.setText("jMenuItem1");
	        minPtsClusterChList.add(minPtsClusterData9);

	        minPtsClusterData10.setText("jMenuItem1");
	        minPtsClusterChList.add(minPtsClusterData10);

	        outputPixelSizeChList.setText("jMenu1");

	        outputPixelSizeData1.setText("jMenuItem1");
	        outputPixelSizeChList.add(outputPixelSizeData1);

	        outputPixelSizeData2.setText("jMenuItem1");
	        outputPixelSizeChList.add(outputPixelSizeData2);

	        outputPixelSizeData3.setText("jMenuItem1");
	        outputPixelSizeChList.add(outputPixelSizeData3);

	        outputPixelSizeData4.setText("jMenuItem1");
	        outputPixelSizeChList.add(outputPixelSizeData4);

	        outputPixelSizeData5.setText("jMenuItem1");
	        outputPixelSizeChList.add(outputPixelSizeData5);

	        outputPixelSizeData6.setText("jMenuItem1");
	        outputPixelSizeChList.add(outputPixelSizeData6);

	        outputPixelSizeData7.setText("jMenuItem1");
	        outputPixelSizeChList.add(outputPixelSizeData7);

	        outputPixelSizeData8.setText("jMenuItem1");
	        outputPixelSizeChList.add(outputPixelSizeData8);

	        outputPixelSizeData9.setText("jMenuItem1");
	        outputPixelSizeChList.add(outputPixelSizeData9);

	        outputPixelSizeData10.setText("jMenuItem1");
	        outputPixelSizeChList.add(outputPixelSizeData10);

	        driftCorrBinLowCountChList.setText("jMenu1");

	        driftCorrBinLowCountData1.setText("jMenuItem1");
	        driftCorrBinLowCountChList.add(driftCorrBinLowCountData1);

	        driftCorrBinLowCountData2.setText("jMenuItem1");
	        driftCorrBinLowCountChList.add(driftCorrBinLowCountData2);

	        driftCorrBinLowCountData3.setText("jMenuItem1");
	        driftCorrBinLowCountChList.add(driftCorrBinLowCountData3);

	        driftCorrBinLowCountData4.setText("jMenuItem1");
	        driftCorrBinLowCountChList.add(driftCorrBinLowCountData4);

	        driftCorrBinLowCountData5.setText("jMenuItem1");
	        driftCorrBinLowCountChList.add(driftCorrBinLowCountData5);

	        driftCorrBinLowCountData6.setText("jMenuItem1");
	        driftCorrBinLowCountChList.add(driftCorrBinLowCountData6);

	        driftCorrBinLowCountData7.setText("jMenuItem1");
	        driftCorrBinLowCountChList.add(driftCorrBinLowCountData7);

	        driftCorrBinLowCountData8.setText("jMenuItem1");
	        driftCorrBinLowCountChList.add(driftCorrBinLowCountData8);

	        driftCorrBinLowCountData9.setText("jMenuItem1");
	        driftCorrBinLowCountChList.add(driftCorrBinLowCountData9);

	        driftCorrBinLowCountData10.setText("jMenuItem1");
	        driftCorrBinLowCountChList.add(driftCorrBinLowCountData10);

	        driftCorrBinHighCountChList.setText("jMenu1");

	        driftCorrBinHighCountData1.setText("jMenuItem1");
	        driftCorrBinHighCountChList.add(driftCorrBinHighCountData1);

	        driftCorrBinHighCountData2.setText("jMenuItem1");
	        driftCorrBinHighCountChList.add(driftCorrBinHighCountData2);

	        driftCorrBinHighCountData3.setText("jMenuItem1");
	        driftCorrBinHighCountChList.add(driftCorrBinHighCountData3);

	        driftCorrBinHighCountData4.setText("jMenuItem1");
	        driftCorrBinHighCountChList.add(driftCorrBinHighCountData4);

	        driftCorrBinHighCountData5.setText("jMenuItem1");
	        driftCorrBinHighCountChList.add(driftCorrBinHighCountData5);

	        driftCorrBinHighCountData6.setText("jMenuItem1");
	        driftCorrBinHighCountChList.add(driftCorrBinHighCountData6);

	        driftCorrBinHighCountData7.setText("jMenuItem1");
	        driftCorrBinHighCountChList.add(driftCorrBinHighCountData7);

	        driftCorrBinHighCountData8.setText("jMenuItem1");
	        driftCorrBinHighCountChList.add(driftCorrBinHighCountData8);

	        driftCorrBinHighCountData9.setText("jMenuItem1");
	        driftCorrBinHighCountChList.add(driftCorrBinHighCountData9);

	        driftCorrBinHighCountData10.setText("jMenuItem1");
	        driftCorrBinHighCountChList.add(driftCorrBinHighCountData10);

	        numberOfBinsDriftCorrChList.setText("jMenu1");

	        numberOfBinsDriftCorrData1.setText("jMenuItem1");
	        numberOfBinsDriftCorrChList.add(numberOfBinsDriftCorrData1);

	        numberOfBinsDriftCorrData2.setText("jMenuItem1");
	        numberOfBinsDriftCorrChList.add(numberOfBinsDriftCorrData2);

	        numberOfBinsDriftCorrData3.setText("jMenuItem1");
	        numberOfBinsDriftCorrChList.add(numberOfBinsDriftCorrData3);

	        numberOfBinsDriftCorrData4.setText("jMenuItem1");
	        numberOfBinsDriftCorrChList.add(numberOfBinsDriftCorrData4);

	        numberOfBinsDriftCorrData5.setText("jMenuItem1");
	        numberOfBinsDriftCorrChList.add(numberOfBinsDriftCorrData5);

	        numberOfBinsDriftCorrData6.setText("jMenuItem1");
	        numberOfBinsDriftCorrChList.add(numberOfBinsDriftCorrData6);

	        numberOfBinsDriftCorrData7.setText("jMenuItem1");
	        numberOfBinsDriftCorrChList.add(numberOfBinsDriftCorrData7);

	        numberOfBinsDriftCorrData8.setText("jMenuItem1");
	        numberOfBinsDriftCorrChList.add(numberOfBinsDriftCorrData8);

	        numberOfBinsDriftCorrData9.setText("jMenuItem1");
	        numberOfBinsDriftCorrChList.add(numberOfBinsDriftCorrData9);

	        numberOfBinsDriftCorrData10.setText("jMenuItem1");
	        numberOfBinsDriftCorrChList.add(numberOfBinsDriftCorrData10);

	        chAlignBinLowCountChList.setText("jMenu1");

	        chAlignBinLowCountData1.setText("jMenuItem1");
	        chAlignBinLowCountChList.add(chAlignBinLowCountData1);

	        chAlignBinLowCountData2.setText("jMenuItem1");
	        chAlignBinLowCountChList.add(chAlignBinLowCountData2);

	        chAlignBinLowCountData3.setText("jMenuItem1");
	        chAlignBinLowCountChList.add(chAlignBinLowCountData3);

	        chAlignBinLowCountData4.setText("jMenuItem1");
	        chAlignBinLowCountChList.add(chAlignBinLowCountData4);

	        chAlignBinLowCountData5.setText("jMenuItem1");
	        chAlignBinLowCountChList.add(chAlignBinLowCountData5);

	        chAlignBinLowCountData6.setText("jMenuItem1");
	        chAlignBinLowCountChList.add(chAlignBinLowCountData6);

	        chAlignBinLowCountData7.setText("jMenuItem1");
	        chAlignBinLowCountChList.add(chAlignBinLowCountData7);

	        chAlignBinLowCountData8.setText("jMenuItem1");
	        chAlignBinLowCountChList.add(chAlignBinLowCountData8);

	        chAlignBinLowCountData9.setText("jMenuItem1");
	        chAlignBinLowCountChList.add(chAlignBinLowCountData9);

	        chAlignBinLowCountData10.setText("jMenuItem1");
	        chAlignBinLowCountChList.add(chAlignBinLowCountData10);

	        chAlignBinHighCountChList.setText("jMenu1");

	        chAlignBinHighCountData1.setText("jMenuItem1");
	        chAlignBinHighCountChList.add(chAlignBinHighCountData1);

	        chAlignBinHighCountData2.setText("jMenuItem1");
	        chAlignBinHighCountChList.add(chAlignBinHighCountData2);

	        chAlignBinHighCountData3.setText("jMenuItem1");
	        chAlignBinHighCountChList.add(chAlignBinHighCountData3);

	        chAlignBinHighCountData4.setText("jMenuItem1");
	        chAlignBinHighCountChList.add(chAlignBinHighCountData4);

	        chAlignBinHighCountData5.setText("jMenuItem1");
	        chAlignBinHighCountChList.add(chAlignBinHighCountData5);

	        chAlignBinHighCountData6.setText("jMenuItem1");
	        chAlignBinHighCountChList.add(chAlignBinHighCountData6);

	        chAlignBinHighCountData7.setText("jMenuItem1");
	        chAlignBinHighCountChList.add(chAlignBinHighCountData7);

	        chAlignBinHighCountData8.setText("jMenuItem1");
	        chAlignBinHighCountChList.add(chAlignBinHighCountData8);

	        chAlignBinHighCountData9.setText("jMenuItem1");
	        chAlignBinHighCountChList.add(chAlignBinHighCountData9);

	        chAlignBinHighCountData10.setText("jMenuItem1");
	        chAlignBinHighCountChList.add(chAlignBinHighCountData10);

	        doPhotonCountChList.setText("jMenu1");

	        doPhotonCountData1.setText("jMenuItem1");
	        doPhotonCountChList.add(doPhotonCountData1);

	        doPhotonCountData2.setText("jMenuItem1");
	        doPhotonCountChList.add(doPhotonCountData2);

	        doPhotonCountData3.setText("jMenuItem1");
	        doPhotonCountChList.add(doPhotonCountData3);

	        doPhotonCountData4.setText("jMenuItem1");
	        doPhotonCountChList.add(doPhotonCountData4);

	        doPhotonCountData5.setText("jMenuItem1");
	        doPhotonCountChList.add(doPhotonCountData5);

	        doPhotonCountData6.setText("jMenuItem1");
	        doPhotonCountChList.add(doPhotonCountData6);

	        doPhotonCountData7.setText("jMenuItem1");
	        doPhotonCountChList.add(doPhotonCountData7);

	        doPhotonCountData8.setText("jMenuItem1");
	        doPhotonCountChList.add(doPhotonCountData8);

	        doPhotonCountData9.setText("jMenuItem1");
	        doPhotonCountChList.add(doPhotonCountData9);

	        doPhotonCountData10.setText("jMenuItem1");
	        doPhotonCountChList.add(doPhotonCountData10);

	        minPhotonCountChList.setText("jMenu1");

	        minPhotonCountData1.setText("jMenuItem1");
	        minPhotonCountChList.add(minPhotonCountData1);

	        minPhotonCountData2.setText("jMenuItem1");
	        minPhotonCountChList.add(minPhotonCountData2);

	        minPhotonCountData3.setText("jMenuItem1");
	        minPhotonCountChList.add(minPhotonCountData3);

	        minPhotonCountData4.setText("jMenuItem1");
	        minPhotonCountChList.add(minPhotonCountData4);

	        minPhotonCountData5.setText("jMenuItem1");
	        minPhotonCountChList.add(minPhotonCountData5);

	        minPhotonCountData6.setText("jMenuItem1");
	        minPhotonCountChList.add(minPhotonCountData6);

	        minPhotonCountData7.setText("jMenuItem1");
	        minPhotonCountChList.add(minPhotonCountData7);

	        minPhotonCountData8.setText("jMenuItem1");
	        minPhotonCountChList.add(minPhotonCountData8);

	        minPhotonCountData9.setText("jMenuItem1");
	        minPhotonCountChList.add(minPhotonCountData9);

	        minPhotonCountData10.setText("jMenuItem1");
	        minPhotonCountChList.add(minPhotonCountData10);

	        maxPhotonCountChList.setText("jMenu1");

	        maxPhotonCountData1.setText("jMenuItem1");
	        maxPhotonCountChList.add(maxPhotonCountData1);

	        maxPhotonCountData2.setText("jMenuItem1");
	        maxPhotonCountChList.add(maxPhotonCountData2);

	        maxPhotonCountData3.setText("jMenuItem1");
	        maxPhotonCountChList.add(maxPhotonCountData3);

	        maxPhotonCountData4.setText("jMenuItem1");
	        maxPhotonCountChList.add(maxPhotonCountData4);

	        maxPhotonCountData5.setText("jMenuItem1");
	        maxPhotonCountChList.add(maxPhotonCountData5);

	        maxPhotonCountData6.setText("jMenuItem1");
	        maxPhotonCountChList.add(maxPhotonCountData6);

	        maxPhotonCountData7.setText("jMenuItem1");
	        maxPhotonCountChList.add(maxPhotonCountData7);

	        maxPhotonCountData8.setText("jMenuItem1");
	        maxPhotonCountChList.add(maxPhotonCountData8);

	        maxPhotonCountData9.setText("jMenuItem1");
	        maxPhotonCountChList.add(maxPhotonCountData9);

	        maxPhotonCountData10.setText("jMenuItem1");
	        maxPhotonCountChList.add(maxPhotonCountData10);

	        doSigmaXYChList.setText("jMenu1");

	        doSigmaXYData1.setText("jMenuItem1");
	        doSigmaXYChList.add(doSigmaXYData1);

	        doSigmaXYData2.setText("jMenuItem1");
	        doSigmaXYChList.add(doSigmaXYData2);

	        doSigmaXYData3.setText("jMenuItem1");
	        doSigmaXYChList.add(doSigmaXYData3);

	        doSigmaXYData4.setText("jMenuItem1");
	        doSigmaXYChList.add(doSigmaXYData4);

	        doSigmaXYData5.setText("jMenuItem1");
	        doSigmaXYChList.add(doSigmaXYData5);

	        doSigmaXYData6.setText("jMenuItem1");
	        doSigmaXYChList.add(doSigmaXYData6);

	        doSigmaXYData7.setText("jMenuItem1");
	        doSigmaXYChList.add(doSigmaXYData7);

	        doSigmaXYData8.setText("jMenuItem1");
	        doSigmaXYChList.add(doSigmaXYData8);

	        doSigmaXYData9.setText("jMenuItem1");
	        doSigmaXYChList.add(doSigmaXYData9);

	        doSigmaXYData10.setText("jMenuItem1");
	        doSigmaXYChList.add(doSigmaXYData10);

	        minSigmaXYChList.setText("jMenu1");

	        minSigmaXYData1.setText("jMenuItem1");
	        minSigmaXYChList.add(minSigmaXYData1);

	        minSigmaXYData2.setText("jMenuItem1");
	        minSigmaXYChList.add(minSigmaXYData2);

	        minSigmaXYData3.setText("jMenuItem1");
	        minSigmaXYChList.add(minSigmaXYData3);

	        minSigmaXYData4.setText("jMenuItem1");
	        minSigmaXYChList.add(minSigmaXYData4);

	        minSigmaXYData5.setText("jMenuItem1");
	        minSigmaXYChList.add(minSigmaXYData5);

	        minSigmaXYData6.setText("jMenuItem1");
	        minSigmaXYChList.add(minSigmaXYData6);

	        minSigmaXYData7.setText("jMenuItem1");
	        minSigmaXYChList.add(minSigmaXYData7);

	        minSigmaXYData8.setText("jMenuItem1");
	        minSigmaXYChList.add(minSigmaXYData8);

	        minSigmaXYData9.setText("jMenuItem1");
	        minSigmaXYChList.add(minSigmaXYData9);

	        minSigmaXYData10.setText("jMenuItem1");
	        minSigmaXYChList.add(minSigmaXYData10);

	        maxSigmaXYChList.setText("jMenu1");

	        maxSigmaXYData1.setText("jMenuItem1");
	        maxSigmaXYChList.add(maxSigmaXYData1);

	        maxSigmaXYData2.setText("jMenuItem1");
	        maxSigmaXYChList.add(maxSigmaXYData2);

	        maxSigmaXYData3.setText("jMenuItem1");
	        maxSigmaXYChList.add(maxSigmaXYData3);

	        maxSigmaXYData4.setText("jMenuItem1");
	        maxSigmaXYChList.add(maxSigmaXYData4);

	        maxSigmaXYData5.setText("jMenuItem1");
	        maxSigmaXYChList.add(maxSigmaXYData5);

	        maxSigmaXYData6.setText("jMenuItem1");
	        maxSigmaXYChList.add(maxSigmaXYData6);

	        maxSigmaXYData7.setText("jMenuItem1");
	        maxSigmaXYChList.add(maxSigmaXYData7);

	        maxSigmaXYData8.setText("jMenuItem1");
	        maxSigmaXYChList.add(maxSigmaXYData8);

	        maxSigmaXYData9.setText("jMenuItem1");
	        maxSigmaXYChList.add(maxSigmaXYData9);

	        maxSigmaXYData10.setText("jMenuItem1");
	        maxSigmaXYChList.add(maxSigmaXYData10);

	        doSigmaZChList.setText("jMenu1");

	        doSigmaZData1.setText("jMenuItem1");
	        doSigmaZChList.add(doSigmaZData1);

	        doSigmaZData2.setText("jMenuItem1");
	        doSigmaZChList.add(doSigmaZData2);

	        doSigmaZData3.setText("jMenuItem1");
	        doSigmaZChList.add(doSigmaZData3);

	        doSigmaZData4.setText("jMenuItem1");
	        doSigmaZChList.add(doSigmaZData4);

	        doSigmaZData5.setText("jMenuItem1");
	        doSigmaZChList.add(doSigmaZData5);

	        doSigmaZData6.setText("jMenuItem1");
	        doSigmaZChList.add(doSigmaZData6);

	        doSigmaZData7.setText("jMenuItem1");
	        doSigmaZChList.add(doSigmaZData7);

	        doSigmaZData8.setText("jMenuItem1");
	        doSigmaZChList.add(doSigmaZData8);

	        doSigmaZData9.setText("jMenuItem1");
	        doSigmaZChList.add(doSigmaZData9);

	        doSigmaZData10.setText("jMenuItem1");
	        doSigmaZChList.add(doSigmaZData10);

	        minSigmaZChList.setText("jMenu1");

	        minSigmaZData1.setText("jMenuItem1");
	        minSigmaZChList.add(minSigmaZData1);

	        minSigmaZData2.setText("jMenuItem1");
	        minSigmaZChList.add(minSigmaZData2);

	        minSigmaZData3.setText("jMenuItem1");
	        minSigmaZChList.add(minSigmaZData3);

	        minSigmaZData4.setText("jMenuItem1");
	        minSigmaZChList.add(minSigmaZData4);

	        minSigmaZData5.setText("jMenuItem1");
	        minSigmaZChList.add(minSigmaZData5);

	        minSigmaZData6.setText("jMenuItem1");
	        minSigmaZChList.add(minSigmaZData6);

	        minSigmaZData7.setText("jMenuItem1");
	        minSigmaZChList.add(minSigmaZData7);

	        minSigmaZData8.setText("jMenuItem1");
	        minSigmaZChList.add(minSigmaZData8);

	        minSigmaZData9.setText("jMenuItem1");
	        minSigmaZChList.add(minSigmaZData9);

	        minSigmaZData10.setText("jMenuItem1");
	        minSigmaZChList.add(minSigmaZData10);

	        maxSigmaZChList.setText("jMenu1");

	        maxSigmaZData1.setText("jMenuItem1");
	        maxSigmaZChList.add(maxSigmaZData1);

	        maxSigmaZData2.setText("jMenuItem1");
	        maxSigmaZChList.add(maxSigmaZData2);

	        maxSigmaZData3.setText("jMenuItem1");
	        maxSigmaZChList.add(maxSigmaZData3);

	        maxSigmaZData4.setText("jMenuItem1");
	        maxSigmaZChList.add(maxSigmaZData4);

	        maxSigmaZData5.setText("jMenuItem1");
	        maxSigmaZChList.add(maxSigmaZData5);

	        maxSigmaZData6.setText("jMenuItem1");
	        maxSigmaZChList.add(maxSigmaZData6);

	        maxSigmaZData7.setText("jMenuItem1");
	        maxSigmaZChList.add(maxSigmaZData7);

	        maxSigmaZData8.setText("jMenuItem1");
	        maxSigmaZChList.add(maxSigmaZData8);

	        maxSigmaZData9.setText("jMenuItem1");
	        maxSigmaZChList.add(maxSigmaZData9);

	        maxSigmaZData10.setText("jMenuItem1");
	        maxSigmaZChList.add(maxSigmaZData10);

	        doRsquareChList.setText("jMenu1");

	        doRsquareData1.setText("jMenuItem1");
	        doRsquareChList.add(doRsquareData1);

	        doRsquareData2.setText("jMenuItem1");
	        doRsquareChList.add(doRsquareData2);

	        doRsquareData3.setText("jMenuItem1");
	        doRsquareChList.add(doRsquareData3);

	        doRsquareData4.setText("jMenuItem1");
	        doRsquareChList.add(doRsquareData4);

	        doRsquareData5.setText("jMenuItem1");
	        doRsquareChList.add(doRsquareData5);

	        doRsquareData6.setText("jMenuItem1");
	        doRsquareChList.add(doRsquareData6);

	        doRsquareData7.setText("jMenuItem1");
	        doRsquareChList.add(doRsquareData7);

	        doRsquareData8.setText("jMenuItem1");
	        doRsquareChList.add(doRsquareData8);

	        doRsquareData9.setText("jMenuItem1");
	        doRsquareChList.add(doRsquareData9);

	        doRsquareData10.setText("jMenuItem1");
	        doRsquareChList.add(doRsquareData10);

	        minRsquareChList.setText("jMenu1");

	        minRsquareData1.setText("jMenuItem1");
	        minRsquareChList.add(minRsquareData1);

	        minRsquareData2.setText("jMenuItem1");
	        minRsquareChList.add(minRsquareData2);

	        minRsquareData3.setText("jMenuItem1");
	        minRsquareChList.add(minRsquareData3);

	        minRsquareData4.setText("jMenuItem1");
	        minRsquareChList.add(minRsquareData4);

	        minRsquareData5.setText("jMenuItem1");
	        minRsquareChList.add(minRsquareData5);

	        minRsquareData6.setText("jMenuItem1");
	        minRsquareChList.add(minRsquareData6);

	        minRsquareData7.setText("jMenuItem1");
	        minRsquareChList.add(minRsquareData7);

	        minRsquareData8.setText("jMenuItem1");
	        minRsquareChList.add(minRsquareData8);

	        minRsquareData9.setText("jMenuItem1");
	        minRsquareChList.add(minRsquareData9);

	        minRsquareData10.setText("jMenuItem1");
	        minRsquareChList.add(minRsquareData10);

	        maxRsquareChList.setText("jMenu1");

	        maxRsquareData1.setText("jMenuItem1");
	        maxRsquareChList.add(maxRsquareData1);

	        maxRsquareData2.setText("jMenuItem1");
	        maxRsquareChList.add(maxRsquareData2);

	        maxRsquareData3.setText("jMenuItem1");
	        maxRsquareChList.add(maxRsquareData3);

	        maxRsquareData4.setText("jMenuItem1");
	        maxRsquareChList.add(maxRsquareData4);

	        maxRsquareData5.setText("jMenuItem1");
	        maxRsquareChList.add(maxRsquareData5);

	        maxRsquareData6.setText("jMenuItem1");
	        maxRsquareChList.add(maxRsquareData6);

	        maxRsquareData7.setText("jMenuItem1");
	        maxRsquareChList.add(maxRsquareData7);

	        maxRsquareData8.setText("jMenuItem1");
	        maxRsquareChList.add(maxRsquareData8);

	        maxRsquareData9.setText("jMenuItem1");
	        maxRsquareChList.add(maxRsquareData9);

	        maxRsquareData10.setText("jMenuItem1");
	        maxRsquareChList.add(maxRsquareData10);

	        doPrecisionXYChList.setText("jMenu1");

	        doPrecisionXYData1.setText("jMenuItem1");
	        doPrecisionXYChList.add(doPrecisionXYData1);

	        doPrecisionXYData2.setText("jMenuItem1");
	        doPrecisionXYChList.add(doPrecisionXYData2);

	        doPrecisionXYData3.setText("jMenuItem1");
	        doPrecisionXYChList.add(doPrecisionXYData3);

	        doPrecisionXYData4.setText("jMenuItem1");
	        doPrecisionXYChList.add(doPrecisionXYData4);

	        doPrecisionXYData5.setText("jMenuItem1");
	        doPrecisionXYChList.add(doPrecisionXYData5);

	        doPrecisionXYData6.setText("jMenuItem1");
	        doPrecisionXYChList.add(doPrecisionXYData6);

	        doPrecisionXYData7.setText("jMenuItem1");
	        doPrecisionXYChList.add(doPrecisionXYData7);

	        doPrecisionXYData8.setText("jMenuItem1");
	        doPrecisionXYChList.add(doPrecisionXYData8);

	        doPrecisionXYData9.setText("jMenuItem1");
	        doPrecisionXYChList.add(doPrecisionXYData9);

	        doPrecisionXYData10.setText("jMenuItem1");
	        doPrecisionXYChList.add(doPrecisionXYData10);

	        minPrecisionXYChList.setText("jMenu1");

	        minPrecisionXYData1.setText("jMenuItem1");
	        minPrecisionXYChList.add(minPrecisionXYData1);

	        minPrecisionXYData2.setText("jMenuItem1");
	        minPrecisionXYChList.add(minPrecisionXYData2);

	        minPrecisionXYData3.setText("jMenuItem1");
	        minPrecisionXYChList.add(minPrecisionXYData3);

	        minPrecisionXYData4.setText("jMenuItem1");
	        minPrecisionXYChList.add(minPrecisionXYData4);

	        minPrecisionXYData5.setText("jMenuItem1");
	        minPrecisionXYChList.add(minPrecisionXYData5);

	        minPrecisionXYData6.setText("jMenuItem1");
	        minPrecisionXYChList.add(minPrecisionXYData6);

	        minPrecisionXYData7.setText("jMenuItem1");
	        minPrecisionXYChList.add(minPrecisionXYData7);

	        minPrecisionXYData8.setText("jMenuItem1");
	        minPrecisionXYChList.add(minPrecisionXYData8);

	        minPrecisionXYData9.setText("jMenuItem1");
	        minPrecisionXYChList.add(minPrecisionXYData9);

	        minPrecisionXYData10.setText("jMenuItem1");
	        minPrecisionXYChList.add(minPrecisionXYData10);

	        maxPrecisionXYChList.setText("jMenu1");

	        maxPrecisionXYData1.setText("jMenuItem1");
	        maxPrecisionXYChList.add(maxPrecisionXYData1);

	        maxPrecisionXYData2.setText("jMenuItem1");
	        maxPrecisionXYChList.add(maxPrecisionXYData2);

	        maxPrecisionXYData3.setText("jMenuItem1");
	        maxPrecisionXYChList.add(maxPrecisionXYData3);

	        maxPrecisionXYData4.setText("jMenuItem1");
	        maxPrecisionXYChList.add(maxPrecisionXYData4);

	        maxPrecisionXYData5.setText("jMenuItem1");
	        maxPrecisionXYChList.add(maxPrecisionXYData5);

	        maxPrecisionXYData6.setText("jMenuItem1");
	        maxPrecisionXYChList.add(maxPrecisionXYData6);

	        maxPrecisionXYData7.setText("jMenuItem1");
	        maxPrecisionXYChList.add(maxPrecisionXYData7);

	        maxPrecisionXYData8.setText("jMenuItem1");
	        maxPrecisionXYChList.add(maxPrecisionXYData8);

	        maxPrecisionXYData9.setText("jMenuItem1");
	        maxPrecisionXYChList.add(maxPrecisionXYData9);

	        maxPrecisionXYData10.setText("jMenuItem1");
	        maxPrecisionXYChList.add(maxPrecisionXYData10);

	        doPrecisionZChList.setText("jMenu1");

	        doPrecisionZData1.setText("jMenuItem1");
	        doPrecisionZChList.add(doPrecisionZData1);

	        doPrecisionZData2.setText("jMenuItem1");
	        doPrecisionZChList.add(doPrecisionZData2);

	        doPrecisionZData3.setText("jMenuItem1");
	        doPrecisionZChList.add(doPrecisionZData3);

	        doPrecisionZData4.setText("jMenuItem1");
	        doPrecisionZChList.add(doPrecisionZData4);

	        doPrecisionZData5.setText("jMenuItem1");
	        doPrecisionZChList.add(doPrecisionZData5);

	        doPrecisionZData6.setText("jMenuItem1");
	        doPrecisionZChList.add(doPrecisionZData6);

	        doPrecisionZData7.setText("jMenuItem1");
	        doPrecisionZChList.add(doPrecisionZData7);

	        doPrecisionZData8.setText("jMenuItem1");
	        doPrecisionZChList.add(doPrecisionZData8);

	        doPrecisionZData9.setText("jMenuItem1");
	        doPrecisionZChList.add(doPrecisionZData9);

	        doPrecisionZData10.setText("jMenuItem1");
	        doPrecisionZChList.add(doPrecisionZData10);

	        minPrecisionZChList.setText("jMenu1");

	        minPrecisionZData1.setText("jMenuItem1");
	        minPrecisionZChList.add(minPrecisionZData1);

	        minPrecisionZData2.setText("jMenuItem1");
	        minPrecisionZChList.add(minPrecisionZData2);

	        minPrecisionZData3.setText("jMenuItem1");
	        minPrecisionZChList.add(minPrecisionZData3);

	        minPrecisionZData4.setText("jMenuItem1");
	        minPrecisionZChList.add(minPrecisionZData4);

	        minPrecisionZData5.setText("jMenuItem1");
	        minPrecisionZChList.add(minPrecisionZData5);

	        minPrecisionZData6.setText("jMenuItem1");
	        minPrecisionZChList.add(minPrecisionZData6);

	        minPrecisionZData7.setText("jMenuItem1");
	        minPrecisionZChList.add(minPrecisionZData7);

	        minPrecisionZData8.setText("jMenuItem1");
	        minPrecisionZChList.add(minPrecisionZData8);

	        minPrecisionZData9.setText("jMenuItem1");
	        minPrecisionZChList.add(minPrecisionZData9);

	        minPrecisionZData10.setText("jMenuItem1");
	        minPrecisionZChList.add(minPrecisionZData10);

	        maxPrecisionZChList.setText("jMenu1");

	        maxPrecisionZData1.setText("jMenuItem1");
	        maxPrecisionZChList.add(maxPrecisionZData1);

	        maxPrecisionZData2.setText("jMenuItem1");
	        maxPrecisionZChList.add(maxPrecisionZData2);

	        maxPrecisionZData3.setText("jMenuItem1");
	        maxPrecisionZChList.add(maxPrecisionZData3);

	        maxPrecisionZData4.setText("jMenuItem1");
	        maxPrecisionZChList.add(maxPrecisionZData4);

	        maxPrecisionZData5.setText("jMenuItem1");
	        maxPrecisionZChList.add(maxPrecisionZData5);

	        maxPrecisionZData6.setText("jMenuItem1");
	        maxPrecisionZChList.add(maxPrecisionZData6);

	        maxPrecisionZData7.setText("jMenuItem1");
	        maxPrecisionZChList.add(maxPrecisionZData7);

	        maxPrecisionZData8.setText("jMenuItem1");
	        maxPrecisionZChList.add(maxPrecisionZData8);

	        maxPrecisionZData9.setText("jMenuItem1");
	        maxPrecisionZChList.add(maxPrecisionZData9);

	        maxPrecisionZData10.setText("jMenuItem1");
	        maxPrecisionZChList.add(maxPrecisionZData10);

	        driftCorrShiftXYChList.setText("jMenu1");

	        driftCorrShiftXYData1.setText("jMenuItem1");
	        driftCorrShiftXYChList.add(driftCorrShiftXYData1);

	        driftCorrShiftXYData2.setText("jMenuItem1");
	        driftCorrShiftXYChList.add(driftCorrShiftXYData2);

	        driftCorrShiftXYData3.setText("jMenuItem1");
	        driftCorrShiftXYChList.add(driftCorrShiftXYData3);

	        driftCorrShiftXYData4.setText("jMenuItem1");
	        driftCorrShiftXYChList.add(driftCorrShiftXYData4);

	        driftCorrShiftXYData5.setText("jMenuItem1");
	        driftCorrShiftXYChList.add(driftCorrShiftXYData5);

	        driftCorrShiftXYData6.setText("jMenuItem1");
	        driftCorrShiftXYChList.add(driftCorrShiftXYData6);

	        driftCorrShiftXYData7.setText("jMenuItem1");
	        driftCorrShiftXYChList.add(driftCorrShiftXYData7);

	        driftCorrShiftXYData8.setText("jMenuItem1");
	        driftCorrShiftXYChList.add(driftCorrShiftXYData8);

	        driftCorrShiftXYData9.setText("jMenuItem1");
	        driftCorrShiftXYChList.add(driftCorrShiftXYData9);

	        driftCorrShiftXYData10.setText("jMenuItem1");
	        driftCorrShiftXYChList.add(driftCorrShiftXYData10);

	        driftCorrShiftZChList.setText("jMenu1");

	        driftCorrShiftZData1.setText("jMenuItem1");
	        driftCorrShiftZChList.add(driftCorrShiftZData1);

	        driftCorrShiftZData2.setText("jMenuItem1");
	        driftCorrShiftZChList.add(driftCorrShiftZData2);

	        driftCorrShiftZData3.setText("jMenuItem1");
	        driftCorrShiftZChList.add(driftCorrShiftZData3);

	        driftCorrShiftZData4.setText("jMenuItem1");
	        driftCorrShiftZChList.add(driftCorrShiftZData4);

	        driftCorrShiftZData5.setText("jMenuItem1");
	        driftCorrShiftZChList.add(driftCorrShiftZData5);

	        driftCorrShiftZData6.setText("jMenuItem1");
	        driftCorrShiftZChList.add(driftCorrShiftZData6);

	        driftCorrShiftZData7.setText("jMenuItem1");
	        driftCorrShiftZChList.add(driftCorrShiftZData7);

	        driftCorrShiftZData8.setText("jMenuItem1");
	        driftCorrShiftZChList.add(driftCorrShiftZData8);

	        driftCorrShiftZData9.setText("jMenuItem1");
	        driftCorrShiftZChList.add(driftCorrShiftZData9);

	        driftCorrShiftZData10.setText("jMenuItem1");
	        driftCorrShiftZChList.add(driftCorrShiftZData10);

	        chAlignShiftXYChList.setText("jMenu1");

	        chAlignShiftXYData1.setText("jMenuItem1");
	        chAlignShiftXYChList.add(chAlignShiftXYData1);

	        chAlignShiftXYData2.setText("jMenuItem1");
	        chAlignShiftXYChList.add(chAlignShiftXYData2);

	        chAlignShiftXYData3.setText("jMenuItem1");
	        chAlignShiftXYChList.add(chAlignShiftXYData3);

	        chAlignShiftXYData4.setText("jMenuItem1");
	        chAlignShiftXYChList.add(chAlignShiftXYData4);

	        chAlignShiftXYData5.setText("jMenuItem1");
	        chAlignShiftXYChList.add(chAlignShiftXYData5);

	        chAlignShiftXYData6.setText("jMenuItem1");
	        chAlignShiftXYChList.add(chAlignShiftXYData6);

	        chAlignShiftXYData7.setText("jMenuItem1");
	        chAlignShiftXYChList.add(chAlignShiftXYData7);

	        chAlignShiftXYData8.setText("jMenuItem1");
	        chAlignShiftXYChList.add(chAlignShiftXYData8);

	        chAlignShiftXYData9.setText("jMenuItem1");
	        chAlignShiftXYChList.add(chAlignShiftXYData9);

	        chAlignShiftXYData10.setText("jMenuItem1");
	        chAlignShiftXYChList.add(chAlignShiftXYData10);

	        chAlignShiftZChList.setText("jMenu1");

	        chAlignShiftZData1.setText("jMenuItem2");
	        chAlignShiftZChList.add(chAlignShiftZData1);

	        chAlignShiftZData2.setText("jMenuItem2");
	        chAlignShiftZChList.add(chAlignShiftZData2);

	        chAlignShiftZData3.setText("jMenuItem2");
	        chAlignShiftZChList.add(chAlignShiftZData3);

	        chAlignShiftZData4.setText("jMenuItem2");
	        chAlignShiftZChList.add(chAlignShiftZData4);

	        chAlignShiftZData5.setText("jMenuItem2");
	        chAlignShiftZChList.add(chAlignShiftZData5);

	        chAlignShiftZData6.setText("jMenuItem2");
	        chAlignShiftZChList.add(chAlignShiftZData6);

	        chAlignShiftZData7.setText("jMenuItem2");
	        chAlignShiftZChList.add(chAlignShiftZData7);

	        chAlignShiftZData8.setText("jMenuItem2");
	        chAlignShiftZChList.add(chAlignShiftZData8);

	        chAlignShiftZData9.setText("jMenuItem2");
	        chAlignShiftZChList.add(chAlignShiftZData9);

	        chAlignShiftZData10.setText("jMenuItem2");
	        chAlignShiftZChList.add(chAlignShiftZData10);

	        doFrameChList.setText("jMenu1");

	        doFrameData1.setText("jMenuItem1");
	        doFrameChList.add(doFrameData1);

	        doFrameData2.setText("jMenuItem1");
	        doFrameChList.add(doFrameData2);

	        doFrameData3.setText("jMenuItem1");
	        doFrameChList.add(doFrameData3);

	        doFrameData4.setText("jMenuItem1");
	        doFrameChList.add(doFrameData4);

	        doFrameData5.setText("jMenuItem1");
	        doFrameChList.add(doFrameData5);

	        doFrameData6.setText("jMenuItem1");
	        doFrameChList.add(doFrameData6);

	        doFrameData7.setText("jMenuItem1");
	        doFrameChList.add(doFrameData7);

	        doFrameData8.setText("jMenuItem1");
	        doFrameChList.add(doFrameData8);

	        doFrameData9.setText("jMenuItem1");
	        doFrameChList.add(doFrameData9);

	        doFrameData10.setText("jMenuItem1");
	        doFrameChList.add(doFrameData10);

	        minFrameChList.setText("jMenu1");

	        minFrameData1.setText("jMenuItem1");
	        minFrameChList.add(minFrameData1);

	        minFrameData2.setText("jMenuItem1");
	        minFrameChList.add(minFrameData2);

	        minFrameData3.setText("jMenuItem1");
	        minFrameChList.add(minFrameData3);

	        minFrameData4.setText("jMenuItem1");
	        minFrameChList.add(minFrameData4);

	        minFrameData5.setText("jMenuItem1");
	        minFrameChList.add(minFrameData5);

	        minFrameData6.setText("jMenuItem1");
	        minFrameChList.add(minFrameData6);

	        minFrameData7.setText("jMenuItem1");
	        minFrameChList.add(minFrameData7);

	        minFrameData8.setText("jMenuItem1");
	        minFrameChList.add(minFrameData8);

	        minFrameData9.setText("jMenuItem1");
	        minFrameChList.add(minFrameData9);

	        minFrameData10.setText("jMenuItem1");
	        minFrameChList.add(minFrameData10);

	        maxFrameChList.setText("jMenu1");

	        maxFrameData1.setText("jMenuItem1");
	        maxFrameChList.add(maxFrameData1);

	        maxFrameData2.setText("jMenuItem1");
	        maxFrameChList.add(maxFrameData2);

	        maxFrameData3.setText("jMenuItem1");
	        maxFrameChList.add(maxFrameData3);

	        maxFrameData4.setText("jMenuItem1");
	        maxFrameChList.add(maxFrameData4);

	        maxFrameData5.setText("jMenuItem1");
	        maxFrameChList.add(maxFrameData5);

	        maxFrameData6.setText("jMenuItem1");
	        maxFrameChList.add(maxFrameData6);

	        maxFrameData7.setText("jMenuItem1");
	        maxFrameChList.add(maxFrameData7);

	        maxFrameData8.setText("jMenuItem1");
	        maxFrameChList.add(maxFrameData8);

	        maxFrameData9.setText("jMenuItem1");
	        maxFrameChList.add(maxFrameData9);

	        maxFrameData10.setText("jMenuItem1");
	        maxFrameChList.add(maxFrameData10);

	        doRenderImageChList.setText("jMenu1");

	        doRenderImageData1.setText("jMenuItem1");
	        doRenderImageChList.add(doRenderImageData1);

	        doRenderImageData2.setText("jMenuItem1");
	        doRenderImageChList.add(doRenderImageData2);

	        doRenderImageData3.setText("jMenuItem1");
	        doRenderImageChList.add(doRenderImageData3);

	        doRenderImageData4.setText("jMenuItem1");
	        doRenderImageChList.add(doRenderImageData4);

	        doRenderImageData5.setText("jMenuItem1");
	        doRenderImageChList.add(doRenderImageData5);

	        doRenderImageData6.setText("jMenuItem1");
	        doRenderImageChList.add(doRenderImageData6);

	        doRenderImageData7.setText("jMenuItem1");
	        doRenderImageChList.add(doRenderImageData7);

	        doRenderImageData8.setText("jMenuItem1");
	        doRenderImageChList.add(doRenderImageData8);

	        doRenderImageData9.setText("jMenuItem1");
	        doRenderImageChList.add(doRenderImageData9);

	        doRenderImageData10.setText("jMenuItem1");
	        doRenderImageChList.add(doRenderImageData10);

	        doCorrelativeChList.setText("jMenu1");

	        doCorrelativeData1.setText("jMenuItem1");
	        doCorrelativeChList.add(doCorrelativeData1);

	        doCorrelativeData2.setText("jMenuItem1");
	        doCorrelativeChList.add(doCorrelativeData2);

	        doCorrelativeData3.setText("jMenuItem1");
	        doCorrelativeChList.add(doCorrelativeData3);

	        doCorrelativeData4.setText("jMenuItem1");
	        doCorrelativeChList.add(doCorrelativeData4);

	        doCorrelativeData5.setText("jMenuItem1");
	        doCorrelativeChList.add(doCorrelativeData5);

	        doCorrelativeData6.setText("jMenuItem1");
	        doCorrelativeChList.add(doCorrelativeData6);

	        doCorrelativeData7.setText("jMenuItem1");
	        doCorrelativeChList.add(doCorrelativeData7);

	        doCorrelativeData8.setText("jMenuItem1");
	        doCorrelativeChList.add(doCorrelativeData8);

	        doCorrelativeData9.setText("jMenuItem1");
	        doCorrelativeChList.add(doCorrelativeData9);

	        doCorrelativeData10.setText("jMenuItem1");
	        doCorrelativeChList.add(doCorrelativeData10);

	        doChromaticChList.setText("jMenu1");

	        doChromaticData1.setText("jMenuItem1");
	        doChromaticChList.add(doChromaticData1);

	        doChromaticData2.setText("jMenuItem1");
	        doChromaticChList.add(doChromaticData2);

	        doChromaticData3.setText("jMenuItem1");
	        doChromaticChList.add(doChromaticData3);

	        doChromaticData4.setText("jMenuItem1");
	        doChromaticChList.add(doChromaticData4);

	        doChromaticData5.setText("jMenuItem1");
	        doChromaticChList.add(doChromaticData5);

	        doChromaticData6.setText("jMenuItem1");
	        doChromaticChList.add(doChromaticData6);

	        doChromaticData7.setText("jMenuItem1");
	        doChromaticChList.add(doChromaticData7);

	        doChromaticData8.setText("jMenuItem1");
	        doChromaticChList.add(doChromaticData8);

	        doChromaticData9.setText("jMenuItem1");
	        doChromaticChList.add(doChromaticData9);

	        doChromaticData10.setText("jMenuItem1");
	        doChromaticChList.add(doChromaticData10);

	        setDefaultCloseOperation(javax.swing.WindowConstants.DISPOSE_ON_CLOSE);
	        setFont(new java.awt.Font("Times New Roman", 0, 12)); // NOI18N

	        Header.setFont(new java.awt.Font("Times New Roman", 3, 24)); // NOI18N
	        Header.setText("SMlocalizer");

	        BasicInp.setBackground(new java.awt.Color(204, 204, 204));
	        BasicInp.setBorder(javax.swing.BorderFactory.createBevelBorder(javax.swing.border.BevelBorder.RAISED, null, null, new java.awt.Color(153, 153, 153), new java.awt.Color(204, 204, 204)));
	        BasicInp.setForeground(new java.awt.Color(153, 153, 153));
	        BasicInp.setToolTipText("Minimum required settings for running SMLocalizer");

	        basicInput.setFont(new java.awt.Font("Times New Roman", 1, 14)); // NOI18N
	        basicInput.setText("Basic input");
	        basicInput.setToolTipText("Minimum required settings for running SMLocalizer");

	        inputPixelSizeLabel.setFont(new java.awt.Font("Times New Roman", 1, 12)); // NOI18N
	        inputPixelSizeLabel.setText("Image pixel size [nm]");
	        inputPixelSizeLabel.setToolTipText("Input pixel size of images");

	        inputPixelSize.setFont(new java.awt.Font("Times New Roman", 0, 12)); // NOI18N
	        inputPixelSize.setHorizontalAlignment(javax.swing.JTextField.CENTER);
	        inputPixelSize.setText("100");
	        inputPixelSize.setToolTipText("Input pixel size of images");

	        totalGainLabel.setFont(new java.awt.Font("Times New Roman", 1, 12)); // NOI18N
	        totalGainLabel.setText("Total gain");
	        totalGainLabel.setToolTipText("Camera specific settings. Total gain from photon to image intensity.");

	        totalGain.setFont(new java.awt.Font("Times New Roman", 0, 12)); // NOI18N
	        totalGain.setHorizontalAlignment(javax.swing.JTextField.CENTER);
	        totalGain.setText("100");
	        totalGain.setToolTipText("Camera specific settings. Total gain from photon to image intensity.");
	        totalGain.addActionListener(new java.awt.event.ActionListener() {
	            public void actionPerformed(java.awt.event.ActionEvent evt) {
	                totalGainActionPerformed(evt);
	            }
	        });

	        minimalSignalLabel.setFont(new java.awt.Font("Times New Roman", 1, 12)); // NOI18N
	        minimalSignalLabel.setText("Minimal signal");
	        minimalSignalLabel.setToolTipText("Miinimal intensity from center pixel required for fitting.");

	        minimalSignal.setFont(new java.awt.Font("Times New Roman", 0, 12)); // NOI18N
	        minimalSignal.setHorizontalAlignment(javax.swing.JTextField.CENTER);
	        minimalSignal.setText("2000");
	        minimalSignal.setToolTipText("Miinimal intensity from center pixel required for fitting.");
	        minimalSignal.addActionListener(new java.awt.event.ActionListener() {
	            public void actionPerformed(java.awt.event.ActionEvent evt) {
	                minimalSignalActionPerformed(evt);
	            }
	        });

	        windowWidthLabel.setFont(new java.awt.Font("Times New Roman", 1, 12)); // NOI18N
	        windowWidthLabel.setText("Filter width [frames]");
	        windowWidthLabel.setToolTipText("Filter window width in frames for time median background filtering.");

	        windowWidth.setFont(new java.awt.Font("Times New Roman", 0, 12)); // NOI18N
	        windowWidth.setHorizontalAlignment(javax.swing.JTextField.CENTER);
	        windowWidth.setText("101");
	        windowWidth.setToolTipText("Filter window width in frames for time median background filtering.");
	        windowWidth.addActionListener(new java.awt.event.ActionListener() {
	            public void actionPerformed(java.awt.event.ActionEvent evt) {
	                windowWidthActionPerformed(evt);
	            }
	        });

	        Process.setFont(new java.awt.Font("Times New Roman", 3, 14)); // NOI18N
	        Process.setText("Process");
	        Process.setToolTipText("Process current image stack including checked algorithms (Render Image, Drift Correct, Align Channels or Cluster analysis). Uses parameter range.");
	        Process.addActionListener(new java.awt.event.ActionListener() {
	            public void actionPerformed(java.awt.event.ActionEvent evt) {
	                ProcessActionPerformed(evt);
	            }
	        });

	        resetBasicInput.setFont(new java.awt.Font("Times New Roman", 3, 12)); // NOI18N
	        resetBasicInput.setText("Reset");
	        resetBasicInput.setToolTipText("Reset Basic input parameters to default.");
	        resetBasicInput.addActionListener(new java.awt.event.ActionListener() {
	            public void actionPerformed(java.awt.event.ActionEvent evt) {
	                resetBasicInputActionPerformed(evt);
	            }
	        });

	        calibrate.setFont(new java.awt.Font("Times New Roman", 3, 12)); // NOI18N
	        calibrate.setText("Calibrate");
	        calibrate.addActionListener(new java.awt.event.ActionListener() {
	            public void actionPerformed(java.awt.event.ActionEvent evt) {
	                calibrateActionPerformed(evt);
	            }
	        });

	        modality.setFont(new java.awt.Font("Times New Roman", 3, 12)); // NOI18N
	        modality.setModel(new javax.swing.DefaultComboBoxModel<>(new String[] { "2D", "PRILM", "Biplane", "Double Helix", "Astigmatism" }));

	        channelId.setFont(new java.awt.Font("Times New Roman", 3, 12)); // NOI18N
	        channelId.setModel(new javax.swing.DefaultComboBoxModel<>(new String[] { "Add channel", "Channel 1" }));
	        channelId.setSelectedIndex(1);
	        channelId.setToolTipText("Select or add new channel for channel specifc settings");
	        channelId.addMouseListener(new java.awt.event.MouseAdapter() {
	            public void mouseClicked(java.awt.event.MouseEvent evt) {
	                channelIdMouseClicked(evt);
	            }
	        });
	        channelId.addActionListener(new java.awt.event.ActionListener() {
	            public void actionPerformed(java.awt.event.ActionEvent evt) {
	                channelIdActionPerformed(evt);
	            }
	        });

	        javax.swing.GroupLayout BasicInpLayout = new javax.swing.GroupLayout(BasicInp);
	        BasicInp.setLayout(BasicInpLayout);
	        BasicInpLayout.setHorizontalGroup(
	            BasicInpLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
	            .addGroup(BasicInpLayout.createSequentialGroup()
	                .addContainerGap()
	                .addGroup(BasicInpLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
	                    .addComponent(Process, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
	                    .addGroup(BasicInpLayout.createSequentialGroup()
	                        .addGroup(BasicInpLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
	                            .addComponent(inputPixelSizeLabel)
	                            .addComponent(totalGainLabel))
	                        .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
	                        .addGroup(BasicInpLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.CENTER)
	                            .addComponent(totalGain)
	                            .addComponent(inputPixelSize)))
	                    .addGroup(javax.swing.GroupLayout.Alignment.TRAILING, BasicInpLayout.createSequentialGroup()
	                        .addComponent(windowWidthLabel)
	                        .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
	                        .addComponent(windowWidth, javax.swing.GroupLayout.DEFAULT_SIZE, 42, Short.MAX_VALUE))
	                    .addGroup(BasicInpLayout.createSequentialGroup()
	                        .addComponent(minimalSignalLabel)
	                        .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
	                        .addComponent(minimalSignal, javax.swing.GroupLayout.PREFERRED_SIZE, 42, javax.swing.GroupLayout.PREFERRED_SIZE))
	                    .addGroup(javax.swing.GroupLayout.Alignment.TRAILING, BasicInpLayout.createSequentialGroup()
	                        .addGroup(BasicInpLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
	                            .addComponent(calibrate, javax.swing.GroupLayout.DEFAULT_SIZE, 87, Short.MAX_VALUE)
	                            .addComponent(resetBasicInput, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE))
	                        .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.UNRELATED)
	                        .addGroup(BasicInpLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING, false)
	                            .addComponent(channelId, 0, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
	                            .addComponent(modality, 0, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)))
	                    .addGroup(javax.swing.GroupLayout.Alignment.TRAILING, BasicInpLayout.createSequentialGroup()
	                        .addGap(0, 0, Short.MAX_VALUE)
	                        .addComponent(basicInput)
	                        .addGap(56, 56, 56)))
	                .addContainerGap())
	        );

	        BasicInpLayout.linkSize(javax.swing.SwingConstants.HORIZONTAL, new java.awt.Component[] {inputPixelSize, minimalSignal, totalGain, windowWidth});

	        BasicInpLayout.setVerticalGroup(
	            BasicInpLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
	            .addGroup(BasicInpLayout.createSequentialGroup()
	                .addContainerGap()
	                .addComponent(basicInput)
	                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
	                .addGroup(BasicInpLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.BASELINE)
	                    .addComponent(inputPixelSizeLabel)
	                    .addComponent(inputPixelSize, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE))
	                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
	                .addGroup(BasicInpLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.BASELINE)
	                    .addComponent(totalGainLabel)
	                    .addComponent(totalGain, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE))
	                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
	                .addGroup(BasicInpLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.BASELINE)
	                    .addComponent(minimalSignalLabel)
	                    .addComponent(minimalSignal, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE))
	                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
	                .addGroup(BasicInpLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.BASELINE)
	                    .addComponent(windowWidth, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
	                    .addComponent(windowWidthLabel))
	                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED, 24, Short.MAX_VALUE)
	                .addGroup(BasicInpLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.BASELINE)
	                    .addComponent(channelId, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
	                    .addComponent(calibrate))
	                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
	                .addGroup(BasicInpLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.BASELINE)
	                    .addComponent(modality, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
	                    .addComponent(resetBasicInput))
	                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
	                .addComponent(Process)
	                .addContainerGap())
	        );

	        ParameterRange.setBorder(javax.swing.BorderFactory.createBevelBorder(javax.swing.border.BevelBorder.RAISED, null, null, new java.awt.Color(204, 204, 204), new java.awt.Color(204, 204, 204)));
	        ParameterRange.setToolTipText("Set the range of selected parameters to include in analysis.");

	        ParameterLabel.setFont(new java.awt.Font("Times New Roman", 1, 14)); // NOI18N
	        ParameterLabel.setText("Parameter range");
	        ParameterLabel.setToolTipText("Set the range of selected parameters to include in analysis.");

	        doPhotonCount.setFont(new java.awt.Font("Times New Roman", 1, 12)); // NOI18N
	        doPhotonCount.setText("Photon count");
	        doPhotonCount.setToolTipText("Include photon count parameter range in selection of what particles from result list to include in analysis.");
	        doPhotonCount.addActionListener(new java.awt.event.ActionListener() {
	            public void actionPerformed(java.awt.event.ActionEvent evt) {
	                doPhotonCountActionPerformed(evt);
	            }
	        });

	        doSigmaXY.setFont(new java.awt.Font("Times New Roman", 1, 12)); // NOI18N
	        doSigmaXY.setSelected(true);
	        doSigmaXY.setText("Sigma x y [nm]");
	        doSigmaXY.setToolTipText("Include sigma x-y parameter range in selection of what particles from result list to include in analysis.");

	        doRsquare.setFont(new java.awt.Font("Times New Roman", 1, 12)); // NOI18N
	        doRsquare.setSelected(true);
	        doRsquare.setText("R^2");
	        doRsquare.setToolTipText("Include R^2 (goodness of fit)  parameter range in selection of what particles from result list to include in analysis.");

	        doPrecisionXY.setFont(new java.awt.Font("Times New Roman", 1, 12)); // NOI18N
	        doPrecisionXY.setText("Precision x y [nm]");
	        doPrecisionXY.setToolTipText("Include precision x-y parameter range in selection of what particles from result list to include in analysis.");

	        doPrecisionZ.setFont(new java.awt.Font("Times New Roman", 1, 12)); // NOI18N
	        doPrecisionZ.setText("Precision z [nm]");
	        doPrecisionZ.setToolTipText("Include precision z parameter range in selection of what particles from result list to include in analysis.");

	        cleanTable.setFont(new java.awt.Font("Times New Roman", 3, 12)); // NOI18N
	        cleanTable.setText("Clean table");
	        cleanTable.setToolTipText("Destructively clean out result table based on selected parameter ranges.");
	        cleanTable.addActionListener(new java.awt.event.ActionListener() {
	            public void actionPerformed(java.awt.event.ActionEvent evt) {
	                cleanTableActionPerformed(evt);
	            }
	        });

	        minPrecisionXY.setFont(new java.awt.Font("Times New Roman", 0, 10)); // NOI18N
	        minPrecisionXY.setHorizontalAlignment(javax.swing.JTextField.CENTER);
	        minPrecisionXY.setText("5");
	        minPrecisionXY.setToolTipText("Include precision x-y parameter range in selection of what particles from result list to include in analysis.");
	        minPrecisionXY.addActionListener(new java.awt.event.ActionListener() {
	            public void actionPerformed(java.awt.event.ActionEvent evt) {
	                minPrecisionXYActionPerformed(evt);
	            }
	        });

	        maxPrecisionXY.setFont(new java.awt.Font("Times New Roman", 0, 10)); // NOI18N
	        maxPrecisionXY.setHorizontalAlignment(javax.swing.JTextField.CENTER);
	        maxPrecisionXY.setText("50");
	        maxPrecisionXY.setToolTipText("Include precision x-y parameter range in selection of what particles from result list to include in analysis.");

	        minRsquare.setFont(new java.awt.Font("Times New Roman", 0, 10)); // NOI18N
	        minRsquare.setHorizontalAlignment(javax.swing.JTextField.CENTER);
	        minRsquare.setText("0.9");
	        minRsquare.setToolTipText("Include R^2 (goodness of fit)  parameter range in selection of what particles from result list to include in analysis.");
	        minRsquare.addActionListener(new java.awt.event.ActionListener() {
	            public void actionPerformed(java.awt.event.ActionEvent evt) {
	                minRsquareActionPerformed(evt);
	            }
	        });

	        maxRsquare.setFont(new java.awt.Font("Times New Roman", 0, 10)); // NOI18N
	        maxRsquare.setHorizontalAlignment(javax.swing.JTextField.CENTER);
	        maxRsquare.setText("1.0");
	        maxRsquare.setToolTipText("Include R^2 (goodness of fit)  parameter range in selection of what particles from result list to include in analysis.");

	        minSigmaXY.setFont(new java.awt.Font("Times New Roman", 0, 10)); // NOI18N
	        minSigmaXY.setHorizontalAlignment(javax.swing.JTextField.CENTER);
	        minSigmaXY.setText("100");
	        minSigmaXY.setToolTipText("Include sigma x-y parameter range in selection of what particles from result list to include in analysis.");
	        minSigmaXY.addActionListener(new java.awt.event.ActionListener() {
	            public void actionPerformed(java.awt.event.ActionEvent evt) {
	                minSigmaXYActionPerformed(evt);
	            }
	        });

	        maxSigmaXY.setFont(new java.awt.Font("Times New Roman", 0, 10)); // NOI18N
	        maxSigmaXY.setHorizontalAlignment(javax.swing.JTextField.CENTER);
	        maxSigmaXY.setText("200");
	        maxSigmaXY.setToolTipText("Include sigma x-y parameter range in selection of what particles from result list to include in analysis.");

	        minPhotonCount.setFont(new java.awt.Font("Times New Roman", 0, 10)); // NOI18N
	        minPhotonCount.setHorizontalAlignment(javax.swing.JTextField.CENTER);
	        minPhotonCount.setText("100");
	        minPhotonCount.setToolTipText("Include photon count parameter range in selection of what particles from result list to include in analysis.");
	        minPhotonCount.addActionListener(new java.awt.event.ActionListener() {
	            public void actionPerformed(java.awt.event.ActionEvent evt) {
	                minPhotonCountActionPerformed(evt);
	            }
	        });

	        maxPhotonCount.setFont(new java.awt.Font("Times New Roman", 0, 10)); // NOI18N
	        maxPhotonCount.setHorizontalAlignment(javax.swing.JTextField.CENTER);
	        maxPhotonCount.setText("5000");
	        maxPhotonCount.setToolTipText("Include photon count parameter range in selection of what particles from result list to include in analysis.");

	        minPrecisionZ.setFont(new java.awt.Font("Times New Roman", 0, 10)); // NOI18N
	        minPrecisionZ.setHorizontalAlignment(javax.swing.JTextField.CENTER);
	        minPrecisionZ.setText("5");
	        minPrecisionZ.setToolTipText("Include precision z parameter range in selection of what particles from result list to include in analysis.");
	        minPrecisionZ.addActionListener(new java.awt.event.ActionListener() {
	            public void actionPerformed(java.awt.event.ActionEvent evt) {
	                minPrecisionZActionPerformed(evt);
	            }
	        });

	        maxPrecisionZ.setFont(new java.awt.Font("Times New Roman", 0, 10)); // NOI18N
	        maxPrecisionZ.setHorizontalAlignment(javax.swing.JTextField.CENTER);
	        maxPrecisionZ.setText("75");
	        maxPrecisionZ.setToolTipText("Include precision z parameter range in selection of what particles from result list to include in analysis.");

	        resetParameterRange.setFont(new java.awt.Font("Times New Roman", 3, 12)); // NOI18N
	        resetParameterRange.setText("Reset");
	        resetParameterRange.setToolTipText("Reset parameter range to default values.");
	        resetParameterRange.addActionListener(new java.awt.event.ActionListener() {
	            public void actionPerformed(java.awt.event.ActionEvent evt) {
	                resetParameterRangeActionPerformed(evt);
	            }
	        });

	        maxLabel3.setFont(new java.awt.Font("Times New Roman", 0, 12)); // NOI18N
	        maxLabel3.setText("max");

	        minLabel3.setFont(new java.awt.Font("Times New Roman", 0, 12)); // NOI18N
	        minLabel3.setText("min");

	        doFrame.setFont(new java.awt.Font("Times New Roman", 1, 12)); // NOI18N
	        doFrame.setText("Frame");
	        doFrame.setToolTipText("Include frames within range in selection of what particles from result list to include in analysis.");
	        doFrame.addActionListener(new java.awt.event.ActionListener() {
	            public void actionPerformed(java.awt.event.ActionEvent evt) {
	                doFrameActionPerformed(evt);
	            }
	        });

	        minFrame.setFont(new java.awt.Font("Times New Roman", 0, 10)); // NOI18N
	        minFrame.setHorizontalAlignment(javax.swing.JTextField.CENTER);
	        minFrame.setText("1");
	        minFrame.setToolTipText("Include frames within range in selection of what particles from result list to include in analysis.");
	        minFrame.addActionListener(new java.awt.event.ActionListener() {
	            public void actionPerformed(java.awt.event.ActionEvent evt) {
	                minFrameActionPerformed(evt);
	            }
	        });

	        maxFrame.setFont(new java.awt.Font("Times New Roman", 0, 10)); // NOI18N
	        maxFrame.setHorizontalAlignment(javax.swing.JTextField.CENTER);
	        maxFrame.setText("100000");
	        maxFrame.setToolTipText("Include frames within range in selection of what particles from result list to include in analysis.");

	        javax.swing.GroupLayout ParameterRangeLayout = new javax.swing.GroupLayout(ParameterRange);
	        ParameterRange.setLayout(ParameterRangeLayout);
	        ParameterRangeLayout.setHorizontalGroup(
	            ParameterRangeLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
	            .addGroup(ParameterRangeLayout.createSequentialGroup()
	                .addContainerGap()
	                .addComponent(ParameterLabel)
	                .addContainerGap(javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE))
	            .addGroup(ParameterRangeLayout.createSequentialGroup()
	                .addGroup(ParameterRangeLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
	                    .addComponent(doPhotonCount)
	                    .addComponent(doSigmaXY)
	                    .addComponent(doRsquare)
	                    .addComponent(doPrecisionXY)
	                    .addComponent(doFrame)
	                    .addGroup(ParameterRangeLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.TRAILING)
	                        .addComponent(cleanTable)
	                        .addComponent(doPrecisionZ)))
	                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED, 22, Short.MAX_VALUE)
	                .addGroup(ParameterRangeLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
	                    .addGroup(javax.swing.GroupLayout.Alignment.TRAILING, ParameterRangeLayout.createSequentialGroup()
	                        .addGroup(ParameterRangeLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.CENTER)
	                            .addComponent(minPhotonCount, javax.swing.GroupLayout.PREFERRED_SIZE, 40, javax.swing.GroupLayout.PREFERRED_SIZE)
	                            .addComponent(minSigmaXY, javax.swing.GroupLayout.PREFERRED_SIZE, 40, javax.swing.GroupLayout.PREFERRED_SIZE)
	                            .addComponent(minRsquare, javax.swing.GroupLayout.PREFERRED_SIZE, 40, javax.swing.GroupLayout.PREFERRED_SIZE)
	                            .addComponent(minPrecisionXY, javax.swing.GroupLayout.PREFERRED_SIZE, 40, javax.swing.GroupLayout.PREFERRED_SIZE)
	                            .addComponent(minPrecisionZ, javax.swing.GroupLayout.PREFERRED_SIZE, 40, javax.swing.GroupLayout.PREFERRED_SIZE)
	                            .addComponent(minFrame, javax.swing.GroupLayout.PREFERRED_SIZE, 40, javax.swing.GroupLayout.PREFERRED_SIZE)
	                            .addComponent(minLabel3))
	                        .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
	                        .addGroup(ParameterRangeLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.CENTER)
	                            .addComponent(maxFrame, javax.swing.GroupLayout.PREFERRED_SIZE, 46, javax.swing.GroupLayout.PREFERRED_SIZE)
	                            .addComponent(maxPrecisionZ, javax.swing.GroupLayout.PREFERRED_SIZE, 46, javax.swing.GroupLayout.PREFERRED_SIZE)
	                            .addComponent(maxPrecisionXY, javax.swing.GroupLayout.PREFERRED_SIZE, 46, javax.swing.GroupLayout.PREFERRED_SIZE)
	                            .addComponent(maxRsquare, javax.swing.GroupLayout.PREFERRED_SIZE, 46, javax.swing.GroupLayout.PREFERRED_SIZE)
	                            .addComponent(maxSigmaXY, javax.swing.GroupLayout.PREFERRED_SIZE, 46, javax.swing.GroupLayout.PREFERRED_SIZE)
	                            .addComponent(maxPhotonCount, javax.swing.GroupLayout.PREFERRED_SIZE, 44, javax.swing.GroupLayout.PREFERRED_SIZE)
	                            .addComponent(maxLabel3))
	                        .addContainerGap())
	                    .addGroup(javax.swing.GroupLayout.Alignment.TRAILING, ParameterRangeLayout.createSequentialGroup()
	                        .addComponent(resetParameterRange, javax.swing.GroupLayout.PREFERRED_SIZE, 93, javax.swing.GroupLayout.PREFERRED_SIZE)
	                        .addGap(18, 18, 18))))
	        );
	        ParameterRangeLayout.setVerticalGroup(
	            ParameterRangeLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
	            .addGroup(ParameterRangeLayout.createSequentialGroup()
	                .addGap(12, 12, 12)
	                .addGroup(ParameterRangeLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.BASELINE)
	                    .addComponent(minLabel3)
	                    .addComponent(maxLabel3)
	                    .addComponent(ParameterLabel))
	                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.UNRELATED)
	                .addGroup(ParameterRangeLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.BASELINE)
	                    .addComponent(minPhotonCount, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
	                    .addComponent(maxPhotonCount, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
	                    .addComponent(doPhotonCount))
	                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
	                .addGroup(ParameterRangeLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.BASELINE)
	                    .addComponent(minSigmaXY, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
	                    .addComponent(maxSigmaXY, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
	                    .addComponent(doSigmaXY))
	                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
	                .addGroup(ParameterRangeLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.BASELINE)
	                    .addComponent(minRsquare, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
	                    .addComponent(maxRsquare, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
	                    .addComponent(doRsquare))
	                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
	                .addGroup(ParameterRangeLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.BASELINE)
	                    .addComponent(minPrecisionXY, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
	                    .addComponent(maxPrecisionXY, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
	                    .addComponent(doPrecisionXY))
	                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
	                .addGroup(ParameterRangeLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.BASELINE)
	                    .addComponent(minPrecisionZ, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
	                    .addComponent(maxPrecisionZ, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
	                    .addComponent(doPrecisionZ))
	                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
	                .addGroup(ParameterRangeLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.BASELINE)
	                    .addComponent(minFrame, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
	                    .addComponent(maxFrame, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
	                    .addComponent(doFrame))
	                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
	                .addGroup(ParameterRangeLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.BASELINE)
	                    .addComponent(resetParameterRange)
	                    .addComponent(cleanTable))
	                .addContainerGap())
	        );

	        Analysis.setBorder(javax.swing.BorderFactory.createBevelBorder(javax.swing.border.BevelBorder.RAISED, null, null, new java.awt.Color(153, 153, 153), new java.awt.Color(204, 204, 204)));

	        doClusterAnalysis.setFont(new java.awt.Font("Times New Roman", 0, 12)); // NOI18N
	        doClusterAnalysis.setToolTipText("Perform cluster analysis during Process execution.");
	        doClusterAnalysis.addActionListener(new java.awt.event.ActionListener() {
	            public void actionPerformed(java.awt.event.ActionEvent evt) {
	                doClusterAnalysisActionPerformed(evt);
	            }
	        });

	        epsilonLabel.setFont(new java.awt.Font("Times New Roman", 1, 12)); // NOI18N
	        epsilonLabel.setText("Epsilon [nm]");
	        epsilonLabel.setToolTipText("Search radius between particles for cluster analysis.");

	        minPtsLabel.setFont(new java.awt.Font("Times New Roman", 1, 12)); // NOI18N
	        minPtsLabel.setText("Minimum connections");
	        minPtsLabel.setToolTipText("Minimum number of connected particles considered a cluster.");

	        epsilon.setFont(new java.awt.Font("Times New Roman", 0, 12)); // NOI18N
	        epsilon.setHorizontalAlignment(javax.swing.JTextField.CENTER);
	        epsilon.setText("10");
	        epsilon.setToolTipText("Search radius between particles for cluster analysis.");

	        minPtsCluster.setFont(new java.awt.Font("Times New Roman", 0, 12)); // NOI18N
	        minPtsCluster.setHorizontalAlignment(javax.swing.JTextField.CENTER);
	        minPtsCluster.setText("5");
	        minPtsCluster.setToolTipText("Minimum number of connected particles considered a cluster.");

	        clusterAnalysis.setFont(new java.awt.Font("Times New Roman", 3, 12)); // NOI18N
	        clusterAnalysis.setText("Cluster analysis");
	        clusterAnalysis.addActionListener(new java.awt.event.ActionListener() {
	            public void actionPerformed(java.awt.event.ActionEvent evt) {
	                clusterAnalysisActionPerformed(evt);
	            }
	        });

	        javax.swing.GroupLayout AnalysisLayout = new javax.swing.GroupLayout(Analysis);
	        Analysis.setLayout(AnalysisLayout);
	        AnalysisLayout.setHorizontalGroup(
	            AnalysisLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
	            .addGroup(AnalysisLayout.createSequentialGroup()
	                .addContainerGap()
	                .addGroup(AnalysisLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
	                    .addGroup(AnalysisLayout.createSequentialGroup()
	                        .addGap(0, 10, Short.MAX_VALUE)
	                        .addGroup(AnalysisLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
	                            .addComponent(minPtsLabel)
	                            .addComponent(epsilonLabel))
	                        .addGap(18, 18, 18)
	                        .addGroup(AnalysisLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
	                            .addComponent(minPtsCluster, javax.swing.GroupLayout.PREFERRED_SIZE, 42, javax.swing.GroupLayout.PREFERRED_SIZE)
	                            .addGroup(javax.swing.GroupLayout.Alignment.TRAILING, AnalysisLayout.createSequentialGroup()
	                                .addComponent(epsilon, javax.swing.GroupLayout.PREFERRED_SIZE, 42, javax.swing.GroupLayout.PREFERRED_SIZE)
	                                .addGap(21, 21, 21))))
	                    .addGroup(AnalysisLayout.createSequentialGroup()
	                        .addComponent(doClusterAnalysis)
	                        .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.UNRELATED)
	                        .addComponent(clusterAnalysis)
	                        .addContainerGap(javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE))))
	        );
	        AnalysisLayout.setVerticalGroup(
	            AnalysisLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
	            .addGroup(AnalysisLayout.createSequentialGroup()
	                .addGap(15, 15, 15)
	                .addGroup(AnalysisLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.TRAILING)
	                    .addComponent(doClusterAnalysis)
	                    .addComponent(clusterAnalysis))
	                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.UNRELATED)
	                .addGroup(AnalysisLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.BASELINE)
	                    .addComponent(epsilon, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
	                    .addComponent(epsilonLabel))
	                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
	                .addGroup(AnalysisLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.BASELINE)
	                    .addComponent(minPtsLabel)
	                    .addComponent(minPtsCluster, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE))
	                .addContainerGap(javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE))
	        );

	        jPanel1.setBorder(javax.swing.BorderFactory.createBevelBorder(javax.swing.border.BevelBorder.RAISED, null, null, new java.awt.Color(153, 153, 153), new java.awt.Color(204, 204, 204)));

	        particlesPerBinLabel.setFont(new java.awt.Font("Times New Roman", 1, 12)); // NOI18N
	        particlesPerBinLabel.setText("Particles per bin");
	        particlesPerBinLabel.setToolTipText("Particles per bin for drift correction.");

	        driftCorrBinLowCount.setFont(new java.awt.Font("Times New Roman", 0, 12)); // NOI18N
	        driftCorrBinLowCount.setHorizontalAlignment(javax.swing.JTextField.CENTER);
	        driftCorrBinLowCount.setText("100");
	        driftCorrBinLowCount.setToolTipText("Particles per bin for drift correction.");

	        driftCorrBinHighCount.setFont(new java.awt.Font("Times New Roman", 0, 12)); // NOI18N
	        driftCorrBinHighCount.setHorizontalAlignment(javax.swing.JTextField.CENTER);
	        driftCorrBinHighCount.setText("1000");
	        driftCorrBinHighCount.setToolTipText("Particles per bin for drift correction.");
	        driftCorrBinHighCount.addActionListener(new java.awt.event.ActionListener() {
	            public void actionPerformed(java.awt.event.ActionEvent evt) {
	                driftCorrBinHighCountActionPerformed(evt);
	            }
	        });

	        numberOfBinsLabel.setFont(new java.awt.Font("Times New Roman", 1, 12)); // NOI18N
	        numberOfBinsLabel.setText("Number of bins");
	        numberOfBinsLabel.setToolTipText("Number of bins to divide the particles in for drift correction.");

	        doDriftCorrect.setSelected(true);
	        doDriftCorrect.setToolTipText("Drift correct particles during Process execution.");
	        doDriftCorrect.addActionListener(new java.awt.event.ActionListener() {
	            public void actionPerformed(java.awt.event.ActionEvent evt) {
	                doDriftCorrectActionPerformed(evt);
	            }
	        });

	        driftCorrect.setFont(new java.awt.Font("Times New Roman", 3, 12)); // NOI18N
	        driftCorrect.setText("Drift correct");
	        driftCorrect.setToolTipText("Drift correct particles in result tabel using parameter range for which to include.");
	        driftCorrect.addActionListener(new java.awt.event.ActionListener() {
	            public void actionPerformed(java.awt.event.ActionEvent evt) {
	                driftCorrectActionPerformed(evt);
	            }
	        });

	        numberOfBinsDriftCorr.setFont(new java.awt.Font("Times New Roman", 0, 12)); // NOI18N
	        numberOfBinsDriftCorr.setHorizontalAlignment(javax.swing.JTextField.CENTER);
	        numberOfBinsDriftCorr.setText("50");
	        numberOfBinsDriftCorr.setToolTipText("Number of bins to divide the particles in for drift correction.");

	        minLabel.setFont(new java.awt.Font("Times New Roman", 0, 12)); // NOI18N
	        minLabel.setText("min");

	        maxLabel.setFont(new java.awt.Font("Times New Roman", 0, 12)); // NOI18N
	        maxLabel.setText("max");

	        particlesPerBinLabel1.setFont(new java.awt.Font("Times New Roman", 1, 12)); // NOI18N
	        particlesPerBinLabel1.setText("Max drift [nm]");
	        particlesPerBinLabel1.setToolTipText("Max drift between two bins. Larger values increase computational time!");

	        driftCorrShiftXY.setFont(new java.awt.Font("Times New Roman", 0, 12)); // NOI18N
	        driftCorrShiftXY.setHorizontalAlignment(javax.swing.JTextField.CENTER);
	        driftCorrShiftXY.setText("250");
	        driftCorrShiftXY.setToolTipText("Max drift between two bins. Larger values increase computational time!");
	        driftCorrShiftXY.addActionListener(new java.awt.event.ActionListener() {
	            public void actionPerformed(java.awt.event.ActionEvent evt) {
	                driftCorrShiftXYActionPerformed(evt);
	            }
	        });

	        driftCorrShiftZ.setFont(new java.awt.Font("Times New Roman", 0, 12)); // NOI18N
	        driftCorrShiftZ.setHorizontalAlignment(javax.swing.JTextField.CENTER);
	        driftCorrShiftZ.setText("250");
	        driftCorrShiftZ.setToolTipText("Max drift between two bins. Larger values increase computational time!");
	        driftCorrShiftZ.addActionListener(new java.awt.event.ActionListener() {
	            public void actionPerformed(java.awt.event.ActionEvent evt) {
	                driftCorrShiftZActionPerformed(evt);
	            }
	        });

	        minLabel2.setFont(new java.awt.Font("Times New Roman", 0, 12)); // NOI18N
	        minLabel2.setText("XY");

	        maxLabel2.setFont(new java.awt.Font("Times New Roman", 0, 12)); // NOI18N
	        maxLabel2.setText("Z");

	        javax.swing.GroupLayout jPanel1Layout = new javax.swing.GroupLayout(jPanel1);
	        jPanel1.setLayout(jPanel1Layout);
	        jPanel1Layout.setHorizontalGroup(
	            jPanel1Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
	            .addGroup(jPanel1Layout.createSequentialGroup()
	                .addContainerGap()
	                .addGroup(jPanel1Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
	                    .addGroup(jPanel1Layout.createSequentialGroup()
	                        .addComponent(doDriftCorrect)
	                        .addGap(22, 22, 22)
	                        .addComponent(driftCorrect, javax.swing.GroupLayout.PREFERRED_SIZE, 108, javax.swing.GroupLayout.PREFERRED_SIZE))
	                    .addComponent(particlesPerBinLabel)
	                    .addGroup(jPanel1Layout.createSequentialGroup()
	                        .addGroup(jPanel1Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.TRAILING)
	                            .addGroup(jPanel1Layout.createSequentialGroup()
	                                .addComponent(numberOfBinsLabel)
	                                .addGap(53, 53, 53)
	                                .addComponent(numberOfBinsDriftCorr, javax.swing.GroupLayout.PREFERRED_SIZE, 33, javax.swing.GroupLayout.PREFERRED_SIZE))
	                            .addGroup(jPanel1Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.TRAILING, false)
	                                .addGroup(jPanel1Layout.createSequentialGroup()
	                                    .addGap(112, 112, 112)
	                                    .addComponent(minLabel)
	                                    .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
	                                    .addComponent(driftCorrBinLowCount, javax.swing.GroupLayout.PREFERRED_SIZE, 32, javax.swing.GroupLayout.PREFERRED_SIZE))
	                                .addGroup(jPanel1Layout.createSequentialGroup()
	                                    .addComponent(particlesPerBinLabel1)
	                                    .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
	                                    .addComponent(minLabel2)
	                                    .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
	                                    .addComponent(driftCorrShiftXY, javax.swing.GroupLayout.PREFERRED_SIZE, 32, javax.swing.GroupLayout.PREFERRED_SIZE))))
	                        .addGap(18, 18, 18)
	                        .addGroup(jPanel1Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
	                            .addComponent(maxLabel)
	                            .addComponent(maxLabel2))
	                        .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
	                        .addGroup(jPanel1Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
	                            .addComponent(driftCorrShiftZ, javax.swing.GroupLayout.PREFERRED_SIZE, 30, javax.swing.GroupLayout.PREFERRED_SIZE)
	                            .addComponent(driftCorrBinHighCount, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE))))
	                .addContainerGap(javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE))
	        );

	        jPanel1Layout.linkSize(javax.swing.SwingConstants.HORIZONTAL, new java.awt.Component[] {driftCorrBinHighCount, driftCorrBinLowCount, driftCorrShiftXY, driftCorrShiftZ, numberOfBinsDriftCorr});

	        jPanel1Layout.setVerticalGroup(
	            jPanel1Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
	            .addGroup(jPanel1Layout.createSequentialGroup()
	                .addContainerGap(javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
	                .addGroup(jPanel1Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
	                    .addComponent(driftCorrect)
	                    .addComponent(doDriftCorrect))
	                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.UNRELATED)
	                .addGroup(jPanel1Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.BASELINE)
	                    .addComponent(particlesPerBinLabel)
	                    .addComponent(driftCorrBinLowCount, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
	                    .addComponent(driftCorrBinHighCount, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
	                    .addComponent(minLabel)
	                    .addComponent(maxLabel))
	                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
	                .addGroup(jPanel1Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.BASELINE)
	                    .addComponent(driftCorrShiftZ, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
	                    .addComponent(maxLabel2)
	                    .addComponent(driftCorrShiftXY, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
	                    .addComponent(minLabel2)
	                    .addComponent(particlesPerBinLabel1))
	                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
	                .addGroup(jPanel1Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.BASELINE)
	                    .addComponent(numberOfBinsDriftCorr, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
	                    .addComponent(numberOfBinsLabel))
	                .addGap(7, 7, 7))
	        );

	        jPanel2.setBorder(javax.swing.BorderFactory.createBevelBorder(javax.swing.border.BevelBorder.RAISED, null, null, new java.awt.Color(153, 153, 153), new java.awt.Color(204, 204, 204)));

	        correctBackground.setFont(new java.awt.Font("Times New Roman", 3, 12)); // NOI18N
	        correctBackground.setText("Correct backgound");
	        correctBackground.setToolTipText("Correct background using pixel by pixel time median. Median window set by Filter width in Basic input.");
	        correctBackground.addActionListener(new java.awt.event.ActionListener() {
	            public void actionPerformed(java.awt.event.ActionEvent evt) {
	                correctBackgroundActionPerformed(evt);
	            }
	        });

	        localize_Fit.setFont(new java.awt.Font("Times New Roman", 3, 12)); // NOI18N
	        localize_Fit.setText("Localize");
	        localize_Fit.setToolTipText("Extract regions of interest and fit against 2D gaussian function. ROI size, minimal signal for inclusion, translation of pixel intensity to photons, minimal number of pixels above background and minimum distance between particles in a frame is set in Basic input.");
	        localize_Fit.addActionListener(new java.awt.event.ActionListener() {
	            public void actionPerformed(java.awt.event.ActionEvent evt) {
	                localize_FitActionPerformed(evt);
	            }
	        });

	        loadSettings.setFont(new java.awt.Font("Times New Roman", 3, 12)); // NOI18N
	        loadSettings.setText("Load settings");
	        loadSettings.setToolTipText("Load previously stored settings (or default to restart).");
	        loadSettings.addActionListener(new java.awt.event.ActionListener() {
	            public void actionPerformed(java.awt.event.ActionEvent evt) {
	                loadSettingsActionPerformed(evt);
	            }
	        });

	        storeSettings.setFont(new java.awt.Font("Times New Roman", 3, 12)); // NOI18N
	        storeSettings.setText("Store settings");
	        storeSettings.setToolTipText("Store the current settings for future processing.");
	        storeSettings.addActionListener(new java.awt.event.ActionListener() {
	            public void actionPerformed(java.awt.event.ActionEvent evt) {
	                storeSettingsActionPerformed(evt);
	            }
	        });

	        javax.swing.GroupLayout jPanel2Layout = new javax.swing.GroupLayout(jPanel2);
	        jPanel2.setLayout(jPanel2Layout);
	        jPanel2Layout.setHorizontalGroup(
	            jPanel2Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
	            .addGroup(jPanel2Layout.createSequentialGroup()
	                .addContainerGap(javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
	                .addComponent(correctBackground, javax.swing.GroupLayout.PREFERRED_SIZE, 125, javax.swing.GroupLayout.PREFERRED_SIZE)
	                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
	                .addComponent(localize_Fit, javax.swing.GroupLayout.PREFERRED_SIZE, 125, javax.swing.GroupLayout.PREFERRED_SIZE)
	                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
	                .addComponent(storeSettings, javax.swing.GroupLayout.PREFERRED_SIZE, 124, javax.swing.GroupLayout.PREFERRED_SIZE)
	                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
	                .addComponent(loadSettings, javax.swing.GroupLayout.PREFERRED_SIZE, 124, javax.swing.GroupLayout.PREFERRED_SIZE)
	                .addContainerGap())
	        );
	        jPanel2Layout.setVerticalGroup(
	            jPanel2Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
	            .addGroup(jPanel2Layout.createSequentialGroup()
	                .addContainerGap()
	                .addGroup(jPanel2Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.BASELINE)
	                    .addComponent(correctBackground)
	                    .addComponent(localize_Fit)
	                    .addComponent(storeSettings)
	                    .addComponent(loadSettings))
	                .addContainerGap(javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE))
	        );

	        jPanel3.setBorder(javax.swing.BorderFactory.createBevelBorder(javax.swing.border.BevelBorder.RAISED, null, null, null, new java.awt.Color(153, 153, 153)));
	        jPanel3.setToolTipText("");

	        renderImage.setFont(new java.awt.Font("Times New Roman", 3, 12)); // NOI18N
	        renderImage.setText("Render image");
	        renderImage.setToolTipText("Render image based on fitted particle result table and selected Parameter ranges.");
	        renderImage.addActionListener(new java.awt.event.ActionListener() {
	            public void actionPerformed(java.awt.event.ActionEvent evt) {
	                renderImageActionPerformed(evt);
	            }
	        });

	        doRenderImage.setSelected(true);
	        doRenderImage.setToolTipText("Render image during Process execution.");

	        outputPixelSizeLabel.setFont(new java.awt.Font("Times New Roman", 1, 12)); // NOI18N
	        outputPixelSizeLabel.setText("Pixel size [nm]");
	        outputPixelSizeLabel.setToolTipText("Rendered image pixel size.");

	        outputPixelSize.setFont(new java.awt.Font("Times New Roman", 0, 12)); // NOI18N
	        outputPixelSize.setHorizontalAlignment(javax.swing.JTextField.CENTER);
	        outputPixelSize.setText("5");
	        outputPixelSize.setToolTipText("Rendered image pixel size.");

	        doGaussianSmoothing.setFont(new java.awt.Font("Times New Roman", 0, 12)); // NOI18N
	        doGaussianSmoothing.setText("Gaussian smoothing");
	        doGaussianSmoothing.setToolTipText("Add a 2 pixel radius gaussian smoothing of the rendered image.");
	        doGaussianSmoothing.addActionListener(new java.awt.event.ActionListener() {
	            public void actionPerformed(java.awt.event.ActionEvent evt) {
	                doGaussianSmoothingActionPerformed(evt);
	            }
	        });

	        outputPixelSizeZ.setFont(new java.awt.Font("Times New Roman", 0, 12)); // NOI18N
	        outputPixelSizeZ.setHorizontalAlignment(javax.swing.JTextField.CENTER);
	        outputPixelSizeZ.setText("5");
	        outputPixelSizeZ.setToolTipText("Rendered image pixel size.");
	        outputPixelSizeZ.addActionListener(new java.awt.event.ActionListener() {
	            public void actionPerformed(java.awt.event.ActionEvent evt) {
	                outputPixelSizeZActionPerformed(evt);
	            }
	        });

	        XYrenderLabel.setFont(new java.awt.Font("Times New Roman", 0, 12)); // NOI18N
	        XYrenderLabel.setText("XY");

	        ZrenderLabel.setFont(new java.awt.Font("Times New Roman", 0, 12)); // NOI18N
	        ZrenderLabel.setText("Z");

	        javax.swing.GroupLayout jPanel3Layout = new javax.swing.GroupLayout(jPanel3);
	        jPanel3.setLayout(jPanel3Layout);
	        jPanel3Layout.setHorizontalGroup(
	            jPanel3Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
	            .addGroup(jPanel3Layout.createSequentialGroup()
	                .addContainerGap()
	                .addGroup(jPanel3Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
	                    .addComponent(doGaussianSmoothing)
	                    .addGroup(jPanel3Layout.createSequentialGroup()
	                        .addGap(21, 21, 21)
	                        .addComponent(outputPixelSizeLabel)))
	                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.UNRELATED, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
	                .addGroup(jPanel3Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.CENTER)
	                    .addComponent(XYrenderLabel)
	                    .addComponent(outputPixelSize, javax.swing.GroupLayout.PREFERRED_SIZE, 26, javax.swing.GroupLayout.PREFERRED_SIZE))
	                .addGap(18, 18, 18)
	                .addGroup(jPanel3Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.CENTER)
	                    .addComponent(outputPixelSizeZ, javax.swing.GroupLayout.PREFERRED_SIZE, 26, javax.swing.GroupLayout.PREFERRED_SIZE)
	                    .addComponent(ZrenderLabel))
	                .addContainerGap())
	            .addGroup(jPanel3Layout.createSequentialGroup()
	                .addGap(6, 6, 6)
	                .addComponent(doRenderImage)
	                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.UNRELATED)
	                .addComponent(renderImage, javax.swing.GroupLayout.PREFERRED_SIZE, 110, javax.swing.GroupLayout.PREFERRED_SIZE)
	                .addContainerGap(javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE))
	        );
	        jPanel3Layout.setVerticalGroup(
	            jPanel3Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
	            .addGroup(jPanel3Layout.createSequentialGroup()
	                .addContainerGap()
	                .addGroup(jPanel3Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
	                    .addComponent(doRenderImage)
	                    .addComponent(renderImage))
	                .addGap(18, 18, 18)
	                .addGroup(jPanel3Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.BASELINE)
	                    .addComponent(doGaussianSmoothing)
	                    .addComponent(XYrenderLabel)
	                    .addComponent(ZrenderLabel))
	                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
	                .addGroup(jPanel3Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.BASELINE)
	                    .addComponent(outputPixelSizeZ, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
	                    .addComponent(outputPixelSize, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
	                    .addComponent(outputPixelSizeLabel))
	                .addGap(18, 18, 18))
	        );

	        buttonGroup2.add(parallelComputation);
	        parallelComputation.setFont(new java.awt.Font("Tahoma", 0, 12)); // NOI18N
	        parallelComputation.setSelected(true);
	        parallelComputation.setText("parallel computation");
	        parallelComputation.setToolTipText("Only CPU bound computation");

	        buttonGroup2.add(GPUcomputation);
	        GPUcomputation.setFont(new java.awt.Font("Tahoma", 0, 12)); // NOI18N
	        GPUcomputation.setText("GPU computation");
	        GPUcomputation.setToolTipText("Transfer bulk of computation to GPU. Not functional on Mac OS.");

	        jPanel4.setBorder(javax.swing.BorderFactory.createBevelBorder(javax.swing.border.BevelBorder.RAISED));

	        doChannelAlign.setToolTipText("Align channels (chromatic shifts) during Process execution.");
	        doChannelAlign.addActionListener(new java.awt.event.ActionListener() {
	            public void actionPerformed(java.awt.event.ActionEvent evt) {
	                doChannelAlignActionPerformed(evt);
	            }
	        });

	        alignChannels.setFont(new java.awt.Font("Times New Roman", 3, 12)); // NOI18N
	        alignChannels.setText("Align channels");
	        alignChannels.setToolTipText("Align channels (chromatic shifts) using the particles included wihtin selected parameter ranges.");
	        alignChannels.addActionListener(new java.awt.event.ActionListener() {
	            public void actionPerformed(java.awt.event.ActionEvent evt) {
	                alignChannelsActionPerformed(evt);
	            }
	        });

	        particlesPerBinLabelchAlign.setFont(new java.awt.Font("Times New Roman", 1, 12)); // NOI18N
	        particlesPerBinLabelchAlign.setText("Particles per bin");
	        particlesPerBinLabelchAlign.setToolTipText("Particles per bin from each chanel for channel alignment.");

	        chAlignBinLowCount.setFont(new java.awt.Font("Times New Roman", 0, 12)); // NOI18N
	        chAlignBinLowCount.setHorizontalAlignment(javax.swing.JTextField.CENTER);
	        chAlignBinLowCount.setText("100");
	        chAlignBinLowCount.setToolTipText("Particles per bin from each chanel for channel alignment.");

	        minLabel1.setFont(new java.awt.Font("Times New Roman", 0, 12)); // NOI18N
	        minLabel1.setText("min");

	        maxLabel1.setFont(new java.awt.Font("Times New Roman", 0, 12)); // NOI18N
	        maxLabel1.setText("max");

	        chAlignBinHighCount.setFont(new java.awt.Font("Times New Roman", 0, 12)); // NOI18N
	        chAlignBinHighCount.setHorizontalAlignment(javax.swing.JTextField.CENTER);
	        chAlignBinHighCount.setText("1000");
	        chAlignBinHighCount.setToolTipText("Particles per bin from each chanel for channel alignment.");
	        chAlignBinHighCount.addActionListener(new java.awt.event.ActionListener() {
	            public void actionPerformed(java.awt.event.ActionEvent evt) {
	                chAlignBinHighCountActionPerformed(evt);
	            }
	        });

	        chAlignShiftZ.setFont(new java.awt.Font("Times New Roman", 0, 12)); // NOI18N
	        chAlignShiftZ.setHorizontalAlignment(javax.swing.JTextField.CENTER);
	        chAlignShiftZ.setText("250");
	        chAlignShiftZ.setToolTipText("Maximal allowed shift between channels. Larger values increase computational time!");
	        chAlignShiftZ.addActionListener(new java.awt.event.ActionListener() {
	            public void actionPerformed(java.awt.event.ActionEvent evt) {
	                chAlignShiftZActionPerformed(evt);
	            }
	        });

	        maxLabel4.setFont(new java.awt.Font("Times New Roman", 0, 12)); // NOI18N
	        maxLabel4.setText("Z");

	        chAlignShiftXY.setFont(new java.awt.Font("Times New Roman", 0, 12)); // NOI18N
	        chAlignShiftXY.setHorizontalAlignment(javax.swing.JTextField.CENTER);
	        chAlignShiftXY.setText("250");
	        chAlignShiftXY.setToolTipText("Maximal allowed shift between channels. Larger values increase computational time!");

	        minLabel4.setFont(new java.awt.Font("Times New Roman", 0, 12)); // NOI18N
	        minLabel4.setText("XY");

	        particlesPerBinLabel2.setFont(new java.awt.Font("Times New Roman", 1, 12)); // NOI18N
	        particlesPerBinLabel2.setText("Max shift [nm]");
	        particlesPerBinLabel2.setToolTipText("Maximal allowed shift between channels. Larger values increase computational time!");

	        javax.swing.GroupLayout jPanel4Layout = new javax.swing.GroupLayout(jPanel4);
	        jPanel4.setLayout(jPanel4Layout);
	        jPanel4Layout.setHorizontalGroup(
	            jPanel4Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
	            .addGroup(jPanel4Layout.createSequentialGroup()
	                .addContainerGap()
	                .addGroup(jPanel4Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
	                    .addGroup(jPanel4Layout.createSequentialGroup()
	                        .addGroup(jPanel4Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
	                            .addGroup(jPanel4Layout.createSequentialGroup()
	                                .addComponent(particlesPerBinLabel2)
	                                .addGap(25, 25, 25))
	                            .addGroup(javax.swing.GroupLayout.Alignment.TRAILING, jPanel4Layout.createSequentialGroup()
	                                .addComponent(particlesPerBinLabelchAlign)
	                                .addGap(18, 18, 18)))
	                        .addGroup(jPanel4Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING, false)
	                            .addComponent(minLabel1)
	                            .addGroup(jPanel4Layout.createSequentialGroup()
	                                .addGap(1, 1, 1)
	                                .addComponent(minLabel4)))
	                        .addGap(10, 10, 10)
	                        .addGroup(jPanel4Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
	                            .addComponent(chAlignBinLowCount, javax.swing.GroupLayout.PREFERRED_SIZE, 34, javax.swing.GroupLayout.PREFERRED_SIZE)
	                            .addComponent(chAlignShiftXY, javax.swing.GroupLayout.PREFERRED_SIZE, 34, javax.swing.GroupLayout.PREFERRED_SIZE))
	                        .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
	                        .addGroup(jPanel4Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
	                            .addComponent(maxLabel1)
	                            .addGroup(jPanel4Layout.createSequentialGroup()
	                                .addGap(3, 3, 3)
	                                .addComponent(maxLabel4)))
	                        .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
	                        .addGroup(jPanel4Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
	                            .addComponent(chAlignShiftZ, javax.swing.GroupLayout.PREFERRED_SIZE, 35, javax.swing.GroupLayout.PREFERRED_SIZE)
	                            .addComponent(chAlignBinHighCount, javax.swing.GroupLayout.PREFERRED_SIZE, 35, javax.swing.GroupLayout.PREFERRED_SIZE)))
	                    .addGroup(jPanel4Layout.createSequentialGroup()
	                        .addComponent(doChannelAlign)
	                        .addGap(18, 18, 18)
	                        .addComponent(alignChannels)))
	                .addContainerGap(javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE))
	        );
	        jPanel4Layout.setVerticalGroup(
	            jPanel4Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
	            .addGroup(jPanel4Layout.createSequentialGroup()
	                .addContainerGap()
	                .addGroup(jPanel4Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.TRAILING)
	                    .addComponent(alignChannels)
	                    .addComponent(doChannelAlign))
	                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.UNRELATED)
	                .addGroup(jPanel4Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.BASELINE)
	                    .addComponent(chAlignBinHighCount, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
	                    .addComponent(minLabel1)
	                    .addComponent(maxLabel1)
	                    .addComponent(chAlignBinLowCount, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
	                    .addComponent(particlesPerBinLabelchAlign))
	                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.UNRELATED)
	                .addGroup(jPanel4Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.CENTER)
	                    .addComponent(chAlignShiftXY, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
	                    .addComponent(chAlignShiftZ, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
	                    .addComponent(minLabel4)
	                    .addComponent(maxLabel4)
	                    .addComponent(particlesPerBinLabel2))
	                .addContainerGap(18, Short.MAX_VALUE))
	        );

	        javax.swing.GroupLayout layout = new javax.swing.GroupLayout(getContentPane());
	        getContentPane().setLayout(layout);
	        layout.setHorizontalGroup(
	            layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
	            .addGroup(layout.createSequentialGroup()
	                .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
	                    .addGroup(layout.createSequentialGroup()
	                        .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
	                            .addGroup(layout.createSequentialGroup()
	                                .addGap(18, 18, 18)
	                                .addComponent(BasicInp, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE))
	                            .addGroup(layout.createSequentialGroup()
	                                .addGap(33, 33, 33)
	                                .addComponent(parallelComputation)))
	                        .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
	                        .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING, false)
	                            .addGroup(layout.createSequentialGroup()
	                                .addComponent(ParameterRange, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
	                                .addGap(6, 6, 6)
	                                .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING, false)
	                                    .addComponent(jPanel1, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
	                                    .addComponent(jPanel4, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE))
	                                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
	                                .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING, false)
	                                    .addComponent(Analysis, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
	                                    .addComponent(jPanel3, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)))
	                            .addGroup(layout.createSequentialGroup()
	                                .addComponent(GPUcomputation)
	                                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
	                                .addComponent(jPanel2, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE))))
	                    .addGroup(layout.createSequentialGroup()
	                        .addGap(455, 455, 455)
	                        .addComponent(Header)))
	                .addContainerGap(11, Short.MAX_VALUE))
	        );
	        layout.setVerticalGroup(
	            layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
	            .addGroup(layout.createSequentialGroup()
	                .addContainerGap()
	                .addComponent(Header)
	                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED, 9, Short.MAX_VALUE)
	                .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING, false)
	                    .addComponent(ParameterRange, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
	                    .addComponent(BasicInp, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
	                    .addGroup(javax.swing.GroupLayout.Alignment.TRAILING, layout.createSequentialGroup()
	                        .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING, false)
	                            .addComponent(jPanel1, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
	                            .addComponent(jPanel3, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE))
	                        .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
	                        .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING, false)
	                            .addComponent(jPanel4, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
	                            .addComponent(Analysis, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE))))
	                .addGap(10, 10, 10)
	                .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
	                    .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.BASELINE)
	                        .addComponent(GPUcomputation)
	                        .addComponent(parallelComputation))
	                    .addComponent(jPanel2, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE))
	                .addContainerGap(javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE))
	        );

	        pack();
	    }

	 private void minimalSignalActionPerformed(java.awt.event.ActionEvent evt) {                                              

	 }                                             

	 private void totalGainActionPerformed(java.awt.event.ActionEvent evt) {                                          


	 }                                         
	 private void doChannelAlignActionPerformed(java.awt.event.ActionEvent evt) {                                               

	 }     
	 private void windowWidthActionPerformed(java.awt.event.ActionEvent evt) {                                            

	 }                                           
	 private void doGaussianSmoothingActionPerformed(java.awt.event.ActionEvent evt) {                                                    

	 } 
	 private void calibrateActionPerformed(java.awt.event.ActionEvent evt) {
		 		 
		 if (modality.getSelectedIndex() == 0)
		 {
			 int[] totalGain 		= getTotalGain();
			 BasicFittingCorrections.calibrate(Integer.parseInt(inputPixelSize.getText()),totalGain);
		 } // 2D.
		 else if(modality.getSelectedIndex() == 1)
		 {
			// PRILM.
			 int zStep = Integer.parseInt(JOptionPane.showInputDialog("z step for calibration file? [nm]",
		                "10"));
			 PRILMfitting.calibrate(Integer.parseInt(inputPixelSize.getText()), zStep);
		 }
		 else if(modality.getSelectedIndex() == 2)
		 {
			 // Biplane.
			 int zStep = Integer.parseInt(JOptionPane.showInputDialog("z step for calibration file? [nm]",
		                "10"));
			 BiplaneFitting.calibrate(Integer.parseInt(inputPixelSize.getText()), zStep);
		 }
		 else if(modality.getSelectedIndex() == 3)
		 {
			 // Double helix.
			 int zStep = Integer.parseInt(JOptionPane.showInputDialog("z step for calibration file? [nm]",
		                "10"));
			 DoubleHelixFitting.calibrate(Integer.parseInt(inputPixelSize.getText()), zStep);
		 }
		 else if(modality.getSelectedIndex() == 4)
		 {
			 // Astigmatism.
			 int zStep = Integer.parseInt(JOptionPane.showInputDialog("z step for calibration file? [nm]",
		                "10"));
			 AstigmatismFitting.calibrate(Integer.parseInt(inputPixelSize.getText()), zStep);
		 }
		 
	 } 
	 private void ProcessActionPerformed(java.awt.event.ActionEvent evt) {                                        
		 /*
		  * Take user settings and run through all analysis steps selected. 
		  */

		 updateList(channelId.getSelectedIndex()-1); // store current settings.
		 int pixelSize = Integer.parseInt(inputPixelSize.getText());
		 
		 int[] totalGain 		= getTotalGain();
		 int[] window 			= getWindowWidth(); // get user input, (W-1)/2.
		 int gWindow 			= 5;
		 //int[] minPixelOverBkgrnd = getMinPixelOverBackground();
		 double maxSigma 		= 2; // 2D 
		 int[] signalStrength 	= getMinSignal();
		 int[] desiredPixelSize = getOutputPixelSize();
		 int selectedModel 		= 5;

		 String modalityChoice 	= "";
		 if (modality.getSelectedIndex() == 0)
		 {
			 modalityChoice = "2D";
				if (pixelSize < 100) // if smaller pixel size the window width needs to increase.
				{
					gWindow = (int) Math.ceil(500 / pixelSize); // 500 nm wide window.

				}
				if (gWindow%2 == 0)
					gWindow++;			 
		 } // 2D.
		 else if(modality.getSelectedIndex() == 1)
		 {
			 modalityChoice = "PRILM";
			 gWindow 		= (int)ij.Prefs.get("SMLocalizer.calibration.PRILM.window", 0);
			 maxSigma 		= (int)ij.Prefs.get("SMLocalizer.calibration.PRILM.sigma", 0);
		 }
		 else if(modality.getSelectedIndex() == 2)
		 {
			 modalityChoice = "Biplane";
			 gWindow 		= (int)ij.Prefs.get("SMLocalizer.calibration.Biplane.window", 0);
			 maxSigma 		= (int)ij.Prefs.get("SMLocalizer.calibration.Biplane.sigma", 0);
		 }
		 else if(modality.getSelectedIndex() == 3)
		 {
			 modalityChoice = "Double Helix";
			 gWindow 		= (int)ij.Prefs.get("SMLocalizer.calibration.DoubleHelix.window", 0);
			 maxSigma 		= (int)ij.Prefs.get("SMLocalizer.calibration.DoubleHelix.sigma", 0);
		 }
		 else if(modality.getSelectedIndex() == 4)
		 {
			 modalityChoice = "Astigmatism";
			 gWindow 		= (int)ij.Prefs.get("SMLocalizer.calibration.Astigmatism.window", 0);
			 maxSigma 		= (int)ij.Prefs.get("SMLocalizer.calibration.Astigmatism.sigma", 0);
		 }


		 if (parallelComputation.isSelected()) // parallel computation.
		 {
			 selectedModel = 0;			
			 
			 BackgroundCorrection.medianFiltering(window,WindowManager.getCurrentImage(),selectedModel); // correct background.			 
			 localizeAndFit.run(signalStrength, gWindow, pixelSize,  totalGain , selectedModel, maxSigma, modalityChoice);  //locate and fit all particles.
			 //			ArrayList<Particle> Results = localizeAndFit.run(signalStrength, minDistance, gWindow, pixelSize,minPixelOverBkgrnd,totalGain,selectedModel);  //locate and fit all particles.
			 //	TableIO.Store(Results);			
		 }
		 else if (GPUcomputation.isSelected()) // GPU accelerated computation.
		 {
			 selectedModel = 2;
			 processMedianFit.run(window, WindowManager.getCurrentImage(), signalStrength, pixelSize, totalGain, maxSigma, gWindow, modalityChoice); // GPU specific call. 
		 }
		
		 boolean[][] include = IncludeParameters();
		 double[][] lb 		= lbParameters();
		 double[][] ub 		= ubParameters();

		 cleanParticleList.run(lb,ub,include);

		 if (doDriftCorrect.isSelected())
		 {
			 int[] minParticles = getDriftCorrBinLowCount();
			 int[] maxParticles = getDriftCorrBinHighCount();
			 int[] bins         = getNumberOfBinsDriftCorr();
			 int[][] boundry    = getDriftCorrShift();            
			 correctDrift.run(boundry, bins, maxParticles, minParticles, selectedModel); // drift correct all channels.
		 }
		 if (doChannelAlign.isSelected())
		 {

			 int[][] boundry = getChAlignShift();
			 int[] minParticles = getChAlignBinLowCount();
			 int[] maxParticles = getChAlignBinHighCount();
			 correctDrift.ChannelAlign(boundry, maxParticles, minParticles,selectedModel); // drift correct all channels.
		 }

		 boolean[] doCluster = getDoClusterAnalysis();
		 double[] epsilon     = getEpsilon();
		 int[] minPts        = getMinPtsCluster();
		 boolean doClusterAnalysis = false;
		 for (int ch = 0; ch < 10; ch++){
			 if (doCluster[ch])
				 doClusterAnalysis = true;
		 }
		 if (doClusterAnalysis)
			 DBClust.Ident(epsilon, minPts,desiredPixelSize,doCluster); // change call to include no loop but checks for number of channels within DBClust.	


		 RenderIm.run(getDoRender(),desiredPixelSize,doGaussianSmoothing.isSelected()); // Not 3D yet

	 }                                       
	 private void doDriftCorrectActionPerformed(java.awt.event.ActionEvent evt) {                                               

	 }   
	 private void doClusterAnalysisActionPerformed(java.awt.event.ActionEvent evt) {                                                  

	 }                                                 
	 private void outputPixelSizeZActionPerformed(java.awt.event.ActionEvent evt) {                                                 

	 }  
	 private void clusterAnalysisActionPerformed(java.awt.event.ActionEvent evt) {                                                
		 updateList(channelId.getSelectedIndex()-1);
		 int[] desiredPixelSize = getOutputPixelSize();
		 boolean[] doCluster = getDoClusterAnalysis();
		 double[] epsilon     = getEpsilon();
		 int[] minPts        = getMinPtsCluster();
		 boolean[][] include = IncludeParameters();
		 double[][] lb = lbParameters();
		 double[][] ub = ubParameters();
		 cleanParticleList.run(lb,ub,include);
		 for (int ch = 1; ch <= 10; ch++)
		 {
			 doCluster[ch-1] = true;
		 }
		 //	nearestNeighbour.analyse();
		 DBClust.Ident(epsilon, minPts,desiredPixelSize,doCluster); // change call to include no loop but checks for number of channels within DBClust.
	 }                                               

	 private void driftCorrBinHighCountActionPerformed(java.awt.event.ActionEvent evt) {                                                      

	 }                                                     

	 private void driftCorrectActionPerformed(java.awt.event.ActionEvent evt) {                                             
		 updateList(channelId.getSelectedIndex()-1);
		 int selectedModel = 5;
		 if (parallelComputation.isSelected()) // parallel computation.
			 selectedModel = 0;
		 else if (GPUcomputation.isSelected()) // GPU accelerated computation.
			 selectedModel = 2;

		 boolean[][] include = IncludeParameters();
		 double[][] lb = lbParameters();
		 double[][] ub = ubParameters();

		 cleanParticleList.run(lb,ub,include);
		 int[] minParticles = getDriftCorrBinLowCount();
		 int[] maxParticles = getDriftCorrBinHighCount();
		 int[] bins         = getNumberOfBinsDriftCorr();
		 int[][] boundry     = getDriftCorrShift();            
		 
		 
		 correctDrift.run(boundry, bins, maxParticles, minParticles, selectedModel); // drift correct all channels.
	 }                                            

	 private void alignChannelsActionPerformed(java.awt.event.ActionEvent evt) {                                              
		 updateList(channelId.getSelectedIndex()-1);
		 int selectedModel = 5;
		 if (parallelComputation.isSelected()) // parallell computation.
			 selectedModel = 0;
		 else if (GPUcomputation.isSelected()) // GPU accelerated computation.
			 selectedModel = 2;

		 boolean[][] include = IncludeParameters();
		 double[][] lb = lbParameters();
		 double[][] ub = ubParameters();
		 cleanParticleList.run(lb,ub,include);
		 int[][] boundry = getChAlignShift();
		 int[] minParticles = getChAlignBinLowCount();
		 int[] maxParticles = getChAlignBinHighCount();
		 correctDrift.ChannelAlign(boundry, maxParticles, minParticles, selectedModel); // drift correct all channels.
	 }                                            

	 private void chAlignBinHighCountActionPerformed(java.awt.event.ActionEvent evt) {                                                    

	 }                                                   

	 private void minPrecisionZActionPerformed(java.awt.event.ActionEvent evt) {                                              

	 }                                             

	 private void minPhotonCountActionPerformed(java.awt.event.ActionEvent evt) {                                               

	 }                                              

	 private void minSigmaXYActionPerformed(java.awt.event.ActionEvent evt) {                                           

	 }                                          
                                    

	 private void minRsquareActionPerformed(java.awt.event.ActionEvent evt) {                                           

	 }                                          

	 private void minPrecisionXYActionPerformed(java.awt.event.ActionEvent evt) {                                               

	 }                                              

	 private void doPhotonCountActionPerformed(java.awt.event.ActionEvent evt) {                                              

	 }                                             

	 private void channelIdActionPerformed(java.awt.event.ActionEvent evt) {                                          
		 int id = channelId.getSelectedIndex();

		 if (id == 0) // if we should add a channel
		 {
			 if (channelId.getItemCount() < 11) // if we have 10 channels already, enough is enough.
			 {

				 channelId.addItem("Channel " + Integer.toString(channelId.getItemCount()));    // add one entry.                                    
				 id = channelId.getItemCount() - 1; // last entry.
				 updateVisible(id-1); // update variables.
				 channelId.setSelectedIndex(id);
			 }
			 else
			 {
				 id = 1;
				 updateVisible(1); // update variables.
				 channelId.setSelectedIndex(id);
			 }
		 }else // change channel.
		 {                    
			 updateVisible(id-1); // update variables.
		 }          
	 }                                         

	 private void channelIdMouseClicked(java.awt.event.MouseEvent evt) {                                       
		 int id = channelId.getSelectedIndex();        
		 updateList(id-1); // store latest variables.
	 }                                      

	 private void resetParameterRangeActionPerformed(java.awt.event.ActionEvent evt) {                                                    
		 //Photon count        
		 doPhotonCount.setSelected(false);                
		 minPhotonCount.setText("100");
		 maxPhotonCount.setText("1000");       
		 // Sigma XY        
		 doSigmaXY.setSelected(false);         
		 minSigmaXY.setText("100");
		 maxSigmaXY.setText("200"); 
		 // Rsquare        
		 doRsquare.setSelected(true);        
		 minRsquare.setText("0.85");
		 maxRsquare.setText("1.0"); 
		 // PrecisionXY
		 doPrecisionXY.setSelected(false);
		 minPrecisionXY.setText("5");
		 maxPrecisionXY.setText("50"); 
		 doPrecisionZ.setSelected(false);
		 minPrecisionZ.setText("5");
		 maxPrecisionZ.setText("75");
		 //frame
		 minFrame.setText("1");
		 maxFrame.setText("100000");
		 updateList(channelId.getSelectedIndex()-1);
	 }                                                   

	 private void resetBasicInputActionPerformed(java.awt.event.ActionEvent evt) {                                                
		 inputPixelSize.setText("100");
		 totalGain.setText("100");
		 minimalSignal.setText("800");
		 windowWidth.setText("101");
		 updateList(channelId.getSelectedIndex()-1); 
	 }                                               

	 private void renderImageActionPerformed(java.awt.event.ActionEvent evt) {      

		 updateList(channelId.getSelectedIndex()-1);        
		 int[] desiredPixelSize = getOutputPixelSize();        
		 boolean[][] include = IncludeParameters();
		 double[][] lb = lbParameters();
		 double[][] ub = ubParameters();        
		 cleanParticleList.run(lb,ub,include);

		 RenderIm.run(getDoRender(),desiredPixelSize,doGaussianSmoothing.isSelected()); // Not 3D yet, how to implement? Need to find out how multi channel images are organized for multi channel functions.
	 }            

	 private void correctBackgroundActionPerformed(java.awt.event.ActionEvent evt) {                                                  
		 updateList(channelId.getSelectedIndex()-1);          
		 int selectedModel = 5;
		 if (parallelComputation.isSelected()) // parallel computation.
			 selectedModel = 0;
		 else if (GPUcomputation.isSelected()) // GPU accelerated computation.
			 selectedModel = 2;


		 int[] Window = getWindowWidth(); // get user input, (W-1)/2.                
		 BackgroundCorrection.medianFiltering(Window,WindowManager.getCurrentImage(),selectedModel); // correct background.		 
	 }                                                 

	 private void localize_FitActionPerformed(java.awt.event.ActionEvent evt) {                                             
		 updateList(channelId.getSelectedIndex()-1);
		 int selectedModel = 5;
		 if (parallelComputation.isSelected()) // parallel computation.
			 selectedModel = 0;
		 else if (GPUcomputation.isSelected()) // GPU accelerated computation.
			 selectedModel = 2;
		
		 int[] totalGain = getTotalGain();        
		 int gWindow = 5;
		 int[] signalStrength = getMinSignal();
		 double maxSigma = 2; // 2D 
		 int pixelSize = Integer.parseInt(inputPixelSize.getText());
		 String modalityChoice = "";
		 if (modality.getSelectedIndex() == 0)
		 {
			 modalityChoice = "2D";
				if (pixelSize < 100) // if smaller pixel size the window width needs to increase.
				{
					gWindow = (int) Math.ceil(500 / pixelSize); // 500 nm wide window.

				}
				if (gWindow%2 == 0)
					gWindow++;			 
		 } // 2D.
		 else if(modality.getSelectedIndex() == 1)
		 {
			 modalityChoice = "PRILM";
			 gWindow 		= (int)ij.Prefs.get("SMLocalizer.calibration.PRILM.window", 0);
			 maxSigma 		= (int)ij.Prefs.get("SMLocalizer.calibration.PRILM.sigma", 0);
		 }
		 else if(modality.getSelectedIndex() == 2)
		 {
			 modalityChoice = "Biplane";
			 gWindow 		= (int)ij.Prefs.get("SMLocalizer.calibration.Biplane.window", 0);
			 maxSigma 		= (int)ij.Prefs.get("SMLocalizer.calibration.Biplane.sigma", 0);
		 }
		 else if(modality.getSelectedIndex() == 3)
		 {
			 modalityChoice = "Double Helix";
			 gWindow 		= (int)ij.Prefs.get("SMLocalizer.calibration.DoubleHelix.window", 0);
			 maxSigma 		= (int)ij.Prefs.get("SMLocalizer.calibration.DoubleHelix.sigma", 0);
		 }
		 else if(modality.getSelectedIndex() == 4)
		 {
			 modalityChoice = "Astigmatism";
			 gWindow 		= (int)ij.Prefs.get("SMLocalizer.calibration.Astigmatism.window", 0);
			 maxSigma 		= (int)ij.Prefs.get("SMLocalizer.calibration.Astigmatism.sigma", 0);
		 }

		 localizeAndFit.run(signalStrength, gWindow, pixelSize,  totalGain , selectedModel, maxSigma, modalityChoice);  //locate and fit all particles.		  	   	  
	 }                                            

	 private void cleanTableActionPerformed(java.awt.event.ActionEvent evt) {                                           
		 updateList(channelId.getSelectedIndex()-1);
		 boolean[][] include = IncludeParameters();
		 double[][] lb = lbParameters();
		 double[][] ub = ubParameters();        
		 cleanParticleList.run(lb,ub,include);
		 cleanParticleList.delete();
	 }                                          

	 private void driftCorrShiftZActionPerformed(java.awt.event.ActionEvent evt) {                                                

	 }                                               

	 private void chAlignShiftZActionPerformed(java.awt.event.ActionEvent evt) {                                              

	 }                                             

	 private void driftCorrShiftXYActionPerformed(java.awt.event.ActionEvent evt) {                                                 

	 }                                                

	 private void minFrameActionPerformed(java.awt.event.ActionEvent evt) {                                         

	 }                                        

	 private void doFrameActionPerformed(java.awt.event.ActionEvent evt) {                                        

	 }                                
	 private void storeSettingsActionPerformed(java.awt.event.ActionEvent evt) {      
		 JPanel panel = new JPanel();
		 panel.add(new JLabel("Please make a selection:"));
		 DefaultComboBoxModel<String> model = new DefaultComboBoxModel<String>();
		 int entries = 0;
		 try 
		 {
			 entries = (int) ij.Prefs.get("SMLocalizer.settingsEntries",0);
		 }finally
		 {

		 }		
		 String[] storeName = new String[entries+1];
		 storeName[0] = "new file";
		 model.addElement(storeName[0] );
		 for (int id = 1; id <= entries; id++){
			 storeName[id] = ij.Prefs.get("SMLocalizer.settingsName"+id, "");
			 model.addElement(storeName[id] );
		 }

		 JComboBox<String> comboBox = new JComboBox<String>(model);
		 panel.add(comboBox);

		 int result = JOptionPane.showConfirmDialog(null, panel, "Settings", JOptionPane.OK_CANCEL_OPTION, JOptionPane.QUESTION_MESSAGE);
		 switch (result) 
		 {
		 case JOptionPane.OK_OPTION:
		 {

			 String loadName = storeName[comboBox.getSelectedIndex()];
			 if(comboBox.getSelectedIndex() == 0) // new name
			 {
				 loadName = JOptionPane.showInputDialog("Settings name?");
				 ij.Prefs.set("SMLocalizer.settingsEntries",entries+1); // adding a new entry
				 ij.Prefs.set("SMLocalizer.settingsName"+(entries+1), loadName);
			 }

			 setParameters(loadName);
		 }             
		 }

	 }                                             

	 private void loadSettingsActionPerformed(java.awt.event.ActionEvent evt) {
		 JPanel panel = new JPanel();
		 panel.add(new JLabel("Please make a selection:"));
		 DefaultComboBoxModel<String> model = new DefaultComboBoxModel<String>();
		 int entries = 0;
		 try 
		 {
			 entries = (int) ij.Prefs.get("SMLocalizer.settingsEntries",0);
		 }finally
		 {

		 }		
		 String[] storeName = new String[entries];
		 for (int id = 1; id <= entries; id++){
			 storeName[id-1] = ij.Prefs.get("SMLocalizer.settingsName"+id, "");
			 model.addElement(storeName[id-1] );
		 }

		 JComboBox<String> comboBox = new JComboBox<String>(model);
		 panel.add(comboBox);

		 int result = JOptionPane.showConfirmDialog(null, panel, "Settings", JOptionPane.OK_CANCEL_OPTION, JOptionPane.QUESTION_MESSAGE);
		 switch (result) {
		 case JOptionPane.OK_OPTION:
		 {
			 loadParameters(storeName[comboBox.getSelectedIndex()]);
		 }                     
		 }       
	 } 

	 /*
	  *   stores current id among the parameters.
	  */
	 private void updateList(int id)
	 {    	
		 checkUserInp(id); // verify that user has input correct values.
		 /*
		  *   Basic input settings
		  */        
		 pixelSizeChList.getItem(id).setText(inputPixelSize.getText());
		 totalGainChList.getItem(id).setText(totalGain.getText());
		 //		minPixelOverBackgroundChList.getItem(id).setText(minPixelOverBackground.getText());
		 minimalSignalChList.getItem(id).setText(minimalSignal.getText());
	//	 gaussWindowChList.getItem(id).setText(String.valueOf(ROIsize.getSelectedIndex()));
		 windowWidthChList.getItem(id).setText(windowWidth.getText());        
		 /*
		  *       Cluster analysis settings.
		  */        
		 if (doClusterAnalysis.isSelected())
			 doClusterAnalysisChList.getItem(id).setText("1");
		 else
			 doClusterAnalysisChList.getItem(id).setText("0");
		 epsilonChList.getItem(id).setText(epsilon.getText());
		 minPtsClusterChList.getItem(id).setText(minPtsCluster.getText());
		 /*
		  *       Render image settings.
		  */ 

		 if (doRenderImage.isSelected())
			 doRenderImageChList.getItem(id).setText("1");
		 else
			 doRenderImageChList.getItem(id).setText("0");

		 //outputPixelSizeChList.getItem(id).setText(outputPixelSize.getText());
		 /*
		  *       Drift and channel correction settings.
		  */        
		 driftCorrBinLowCountChList.getItem(id).setText(driftCorrBinLowCount.getText());
		 driftCorrBinHighCountChList.getItem(id).setText(driftCorrBinHighCount.getText());
		 driftCorrShiftXYChList.getItem(id).setText(driftCorrShiftXY.getText());
		 driftCorrShiftZChList.getItem(id).setText(driftCorrShiftZ.getText());        
		 numberOfBinsDriftCorrChList.getItem(id).setText(numberOfBinsDriftCorr.getText());
		 chAlignBinLowCountChList.getItem(id).setText(chAlignBinLowCount.getText());
		 chAlignBinHighCountChList.getItem(id).setText(chAlignBinHighCount.getText());
		 chAlignShiftXYChList.getItem(id).setText(chAlignShiftXY.getText());
		 chAlignShiftZChList.getItem(id).setText(chAlignShiftZ.getText());
		 		
	
		 /*
		  *   Parameter settings
		  */
		 //Photon count
		 if (doPhotonCount.isSelected())
			 doPhotonCountChList.getItem(id).setText("1");
		 else
			 doPhotonCountChList.getItem(id).setText("0");
		 minPhotonCountChList.getItem(id).setText(minPhotonCount.getText());
		 maxPhotonCountChList.getItem(id).setText(maxPhotonCount.getText());    
		 // Sigma XY
		 if (doSigmaXY.isSelected())
			 doSigmaXYChList.getItem(id).setText("1");
		 else
			 doSigmaXYChList.getItem(id).setText("0");
		 minSigmaXYChList.getItem(id).setText(minSigmaXY.getText());
		 maxSigmaXYChList.getItem(id).setText(maxSigmaXY.getText());  
		 // Sigma Z
		/* if (doSigmaZ.isSelected())
			 doSigmaZChList.getItem(id).setText("1");
		 else
			 doSigmaZChList.getItem(id).setText("0");
		 minSigmaZChList.getItem(id).setText(minSigmaZ.getText());
		 maxSigmaZChList.getItem(id).setText(maxSigmaZ.getText());          */
		 // Rsquare
		 if (doRsquare.isSelected())
			 doRsquareChList.getItem(id).setText("1");
		 else
			 doRsquareChList.getItem(id).setText("0");
		 minRsquareChList.getItem(id).setText(minRsquare.getText());
		 maxRsquareChList.getItem(id).setText(maxRsquare.getText());         
		 // PrecisionXY
		 if (doPrecisionXY.isSelected())
			 doPrecisionXYChList.getItem(id).setText("1");
		 else
			 doPrecisionXYChList.getItem(id).setText("0");
		 minPrecisionXYChList.getItem(id).setText(minPrecisionXY.getText());
		 maxPrecisionXYChList.getItem(id).setText(maxPrecisionXY.getText());         
		 // PrecisionZ
		 if (doPrecisionZ.isSelected())
			 doPrecisionZChList.getItem(id).setText("1");
		 else
			 doPrecisionZChList.getItem(id).setText("0");
		 minPrecisionZChList.getItem(id).setText(minPrecisionZ.getText());
		 maxPrecisionZChList.getItem(id).setText(maxPrecisionZ.getText());

		 // Frame
		 if (doFrame.isSelected())
			 doFrameChList.getItem(id).setText("1");
		 else
			 doFrameChList.getItem(id).setText("0");
		 minFrameChList.getItem(id).setText(minFrame.getText());
		 maxFrameChList.getItem(id).setText(maxFrame.getText());       
	 }

	 /*
	  * update visible datafields for user based on channel selection. id from 0 to 9 for channel 1 to 10.
	  */
	 private void updateVisible(int id)
	 {
		 /*
		  *   Basic input settings
		  */
		 inputPixelSize.setText(pixelSizeChList.getItem(id).getText());
		 totalGain.setText(totalGainChList.getItem(id).getText());
		 //		minPixelOverBackground.setText(minPixelOverBackgroundChList.getItem(id).getText());
		 minimalSignal.setText(minimalSignalChList.getItem(id).getText());
		// ROIsize.setSelectedIndex(Integer.parseInt(gaussWindowChList.getItem(id).getText()));
		 windowWidth.setText( windowWidthChList.getItem(id).getText());
		 /*
		  *       Cluster analysis settings.
		  */
		 if (doClusterAnalysisChList.getItem(id).getText().equals("1"))
			 doClusterAnalysis.setSelected(true);
		 else
			 doClusterAnalysis.setSelected(false);
		 epsilon.setText(epsilonChList.getItem(id).getText());
		 minPtsCluster.setText(minPtsClusterChList.getItem(id).getText());        

		 /*
		  *       Render image settings.
		  */
		 if (doRenderImageChList.getItem(id).getText().equals("1"))
			 doRenderImage.setSelected(true);
		 else
			 doRenderImage.setSelected(false);

		 //outputPixelSize.setText(outputPixelSizeChList.getItem(id).getText());


		 /*
		  *       Drift and channel correction settings.
		  */
		 driftCorrBinLowCount.setText(driftCorrBinLowCountChList.getItem(id).getText());
		 driftCorrBinHighCount.setText(driftCorrBinHighCountChList.getItem(id).getText());
		 driftCorrShiftXY.setText(driftCorrShiftXYChList.getItem(id).getText());
		 driftCorrShiftZ.setText(driftCorrShiftZChList.getItem(id).getText());
		 numberOfBinsDriftCorr.setText(numberOfBinsDriftCorrChList.getItem(id).getText());
		 chAlignBinLowCount.setText(chAlignBinLowCountChList.getItem(id).getText());
		 chAlignBinHighCount.setText(chAlignBinHighCountChList.getItem(id).getText());
		 chAlignShiftXY.setText(chAlignShiftXYChList.getItem(id).getText());
		 chAlignShiftZ.setText(chAlignShiftZChList.getItem(id).getText());
			 
		 
		 /*
		  *   Parameter settings
		  */
		 //Photon count

		 if(doPhotonCountChList.getItem(id).getText().equals("1"))
			 doPhotonCount.setSelected(true);
		 else
			 doPhotonCount.setSelected(false);
		 minPhotonCount.setText(minPhotonCountChList.getItem(id).getText());
		 maxPhotonCount.setText(maxPhotonCountChList.getItem(id).getText());       
		 // Sigma XY
		 if(doSigmaXYChList.getItem(id).getText().equals("1"))
			 doSigmaXY.setSelected(true);
		 else
			 doSigmaXY.setSelected(false);
		 minSigmaXY.setText(minSigmaXYChList.getItem(id).getText());
		 maxSigmaXY.setText(maxSigmaXYChList.getItem(id).getText()); 
		 // Sigma Z
	/*	 if(doSigmaZChList.getItem(id).getText().equals("1"))
			 doSigmaZ.setSelected(true);
		 else
			 doSigmaZ.setSelected(false);
		 minSigmaZ.setText(minSigmaZChList.getItem(id).getText());
		 maxSigmaZ.setText(maxSigmaZChList.getItem(id).getText()); */
		 // Rsquare
		 if(doRsquareChList.getItem(id).getText().equals("1"))
			 doRsquare.setSelected(true);
		 else
			 doRsquare.setSelected(false);
		 minRsquare.setText(minRsquareChList.getItem(id).getText());
		 maxRsquare.setText(maxRsquareChList.getItem(id).getText()); 
		 // PrecisionXY
		 if(doPrecisionXYChList.getItem(id).getText().equals("1"))
			 doPrecisionXY.setSelected(true);
		 else
			 doPrecisionXY.setSelected(false);
		 minPrecisionXY.setText(minPrecisionXYChList.getItem(id).getText());
		 maxPrecisionXY.setText(maxPrecisionXYChList.getItem(id).getText()); 
		 // PrecisionZ
		 if(doPrecisionZChList.getItem(id).getText().equals("1"))
			 doPrecisionZ.setSelected(true);
		 else
			 doPrecisionZ.setSelected(false);
		 minPrecisionZ.setText(minPrecisionZChList.getItem(id).getText());
		 maxPrecisionZ.setText(maxPrecisionZChList.getItem(id).getText());             
		 // Frame
		 if(doFrameChList.getItem(id).getText().equals("1"))
			 doFrame.setSelected(true);
		 else
			 doFrame.setSelected(false);
		 minFrame.setText(minFrameChList.getItem(id).getText());
		 maxFrame.setText(maxFrameChList.getItem(id).getText());             
	 }

	 /*
	  *       Check that user has not tried to set incorrect settings (text instead of numbers etc)
	  */
	 private void checkUserInp(int id)
	 {
		 /*
		  *   Basic input settings
		  */

		 try {
			 int I = Integer.parseInt(inputPixelSize.getText());        		
			 if (I <= 0)
				 inputPixelSize.setText(pixelSizeChList.getItem(id).getText()); 		// Update.                    
		 } catch (NumberFormatException e) { 								// If user wrote non numerical test into the field.                
			 inputPixelSize.setText(pixelSizeChList.getItem(id).getText()); 		// Update.
		 }
		 try {
			 int I = Integer.parseInt(totalGain.getText());        		
			 if (I <= 0)
				 totalGain.setText(totalGainChList.getItem(id).getText()); 		// Update.                    
		 } catch (NumberFormatException e) { 								// If user wrote non numerical test into the field.                
			 totalGain.setText(totalGainChList.getItem(id).getText()); 		// Update.
		 }
		 /*		try {
			int I = Integer.parseInt(minPixelOverBackground.getText());        		
			if (I <= 0)
				minPixelOverBackground.setText(minPixelOverBackgroundChList.getItem(id).getText()); 		// Update.                    
		} catch (NumberFormatException e) { 								// If user wrote non numerical test into the field.                
			minPixelOverBackground.setText(minPixelOverBackgroundChList.getItem(id).getText()); 		// Update.
		}*/
		 try {
			 int I = Integer.parseInt(minimalSignal.getText());        		
			 if (I <= 0)
				 minimalSignal.setText(minimalSignalChList.getItem(id).getText()); 		// Update.                    
		 } catch (NumberFormatException e) { 								// If user wrote non numerical test into the field.                
			 minimalSignal.setText(minimalSignalChList.getItem(id).getText()); 		// Update.
		 }
		 try {
			 int I = Integer.parseInt(windowWidth.getText());        		
			 if (I <= 0)
				 windowWidth.setText(windowWidthChList.getItem(id).getText()); 		// Update.                    
		 } catch (NumberFormatException e) { 								// If user wrote non numerical test into the field.                
			 windowWidth.setText(windowWidthChList.getItem(id).getText()); 		// Update.
		 }
		 /*
		  *       Cluster analysis settings.
		  */
		 try {
			 int I = Integer.parseInt(epsilon.getText());        		
			 if (I <= 0)
				 epsilon.setText(epsilonChList.getItem(id).getText()); 		// Update.                    
		 } catch (NumberFormatException e) { 								// If user wrote non numerical test into the field.                
			 epsilon.setText(epsilonChList.getItem(id).getText()); 		// Update.
		 }
		 try {
			 int I = Integer.parseInt(minPtsCluster.getText());        		
			 if (I <= 0)
				 minPtsCluster.setText(minPtsClusterChList.getItem(id).getText()); 		// Update.                    
		 } catch (NumberFormatException e) { 								// If user wrote non numerical test into the field.                
			 minPtsCluster.setText(minPtsClusterChList.getItem(id).getText()); 		// Update.
		 }                    
		 /*
		  *       Render image settings.
		  */
		 try {
			 int I = Integer.parseInt(outputPixelSize.getText());                 
			 if (I <= 0)
				 outputPixelSize.setText("5"); 		// Update.                    
		 } catch (NumberFormatException e) { 								// If user wrote non numerical test into the field.                
			 outputPixelSize.setText("5"); 		// Update.
		 } 
		 try {
			 int I = Integer.parseInt(outputPixelSizeZ.getText());                 
			 if (I <= 0)
				 outputPixelSizeZ.setText("10"); 		// Update.                    
		 } catch (NumberFormatException e) { 								// If user wrote non numerical test into the field.                
			 outputPixelSizeZ.setText("10"); 		// Update.
		 } 

		 /*
		  *       Drift and channel correction settings.
		  */
		 try {
			 int I = Integer.parseInt(driftCorrBinLowCount.getText());        		
			 if (I <= 0)
				 driftCorrBinLowCount.setText(driftCorrBinLowCountChList.getItem(id).getText()); 		// Update.                    
		 } catch (NumberFormatException e) { 								// If user wrote non numerical test into the field.                
			 driftCorrBinLowCount.setText(driftCorrBinLowCountChList.getItem(id).getText()); 		// Update.
		 }
		 try {
			 int I = Integer.parseInt(driftCorrBinHighCount.getText());        		
			 if (I <= Integer.parseInt(driftCorrBinLowCount.getText()))
				 driftCorrBinHighCount.setText(driftCorrBinLowCount.getText());   
			 if (I <= 0)
				 driftCorrBinHighCount.setText(driftCorrBinHighCountChList.getItem(id).getText()); 		// Update.                    
		 } catch (NumberFormatException e) { 								// If user wrote non numerical test into the field.                
			 driftCorrBinHighCount.setText(driftCorrBinHighCountChList.getItem(id).getText()); 		// Update.
		 }
		 try {
			 int I = Integer.parseInt(driftCorrShiftXY.getText());        		
			 if (I <= 0)
				 driftCorrShiftXY.setText(driftCorrShiftXYChList.getItem(id).getText()); 		// Update.                    
		 } catch (NumberFormatException e) { 								// If user wrote non numerical test into the field.                
			 driftCorrShiftXY.setText(driftCorrShiftXYChList.getItem(id).getText()); 		// Update.
		 }
		 try {
			 int I = Integer.parseInt(driftCorrShiftZ.getText());        		            
			 if (I <= 0)
				 driftCorrShiftZ.setText(driftCorrShiftZChList.getItem(id).getText()); 		// Update.                    
		 } catch (NumberFormatException e) { 								// If user wrote non numerical test into the field.                
			 driftCorrShiftZ.setText(driftCorrShiftZChList.getItem(id).getText()); 		// Update.
		 }        

		 try {
			 int I = Integer.parseInt(numberOfBinsDriftCorr.getText());        		
			 if (I <= 0)
				 numberOfBinsDriftCorr.setText(numberOfBinsDriftCorrChList.getItem(id).getText()); 		// Update.                    
		 } catch (NumberFormatException e) { 								// If user wrote non numerical test into the field.                
			 numberOfBinsDriftCorr.setText(numberOfBinsDriftCorrChList.getItem(id).getText()); 		// Update.
		 }
		 try {
			 int I = Integer.parseInt(chAlignBinLowCount.getText());        		
			 if (I <= 0)
				 chAlignBinLowCount.setText(chAlignBinLowCountChList.getItem(id).getText()); 		// Update.                    
		 } catch (NumberFormatException e) { 								// If user wrote non numerical test into the field.                
			 chAlignBinLowCount.setText(chAlignBinLowCountChList.getItem(id).getText()); 		// Update.
		 }
		 try {
			 int I = Integer.parseInt(chAlignBinHighCount.getText());        		
			 if (I <= Integer.parseInt(chAlignBinLowCount.getText()))
				 chAlignBinHighCount.setText(chAlignBinLowCount.getText());   
			 if (I <= 0)
				 chAlignBinHighCount.setText(chAlignBinHighCountChList.getItem(id).getText()); 		// Update.                    
		 } catch (NumberFormatException e) { 								// If user wrote non numerical test into the field.                
			 chAlignBinHighCount.setText(chAlignBinHighCountChList.getItem(id).getText()); 		// Update.
		 }   
		 try {
			 int I = Integer.parseInt(chAlignShiftXY.getText());        		
			 if (I <= 0)
				 chAlignShiftXY.setText(chAlignShiftXYChList.getItem(id).getText()); 		// Update.                    
		 } catch (NumberFormatException e) { 								// If user wrote non numerical test into the field.                
			 chAlignShiftXY.setText(chAlignShiftXYChList.getItem(id).getText()); 		// Update.
		 }
		 try {
			 int I = Integer.parseInt(chAlignShiftZ.getText());        		            
			 if (I <= 0)
				 chAlignShiftZ.setText(chAlignShiftZChList.getItem(id).getText()); 		// Update.                    
		 } catch (NumberFormatException e) { 								// If user wrote non numerical test into the field.                
			 chAlignShiftZ.setText(chAlignShiftZChList.getItem(id).getText()); 		// Update.
		 }  

		 /*
		  *   Parameter settings
		  */
		 //Photon count
		 try {
			 int I = Integer.parseInt(minPhotonCount.getText());        		
			 if (I <= 0)
				 minPhotonCount.setText(minPhotonCountChList.getItem(id).getText()); 		// Update.                    
		 } catch (NumberFormatException e) { 								// If user wrote non numerical test into the field.                
			 minPhotonCount.setText(minPhotonCountChList.getItem(id).getText()); 		// Update.
		 }
		 try {
			 int I = Integer.parseInt(maxPhotonCount.getText());        		
			 if (I <= Integer.parseInt(minPhotonCount.getText()))
				 maxPhotonCount.setText(minPhotonCount.getText());   
			 if (I <= 0)
				 maxPhotonCount.setText(maxPhotonCountChList.getItem(id).getText()); 		// Update.                    
		 } catch (NumberFormatException e) { 								// If user wrote non numerical test into the field.                
			 maxPhotonCount.setText(maxPhotonCountChList.getItem(id).getText()); 		// Update.
		 }            
		 // Sigma XY
		 try {
			 int I = Integer.parseInt(minSigmaXY.getText());        		
			 if (I <= 0)
				 minSigmaXY.setText(minSigmaXYChList.getItem(id).getText()); 		// Update.                    
		 } catch (NumberFormatException e) { 								// If user wrote non numerical test into the field.                
			 minSigmaXY.setText(minSigmaXYChList.getItem(id).getText()); 		// Update.
		 }
		 try {
			 int I = Integer.parseInt(maxSigmaXY.getText());        		
			 if (I <= Integer.parseInt(minSigmaXY.getText()))
				 maxSigmaXY.setText(minSigmaXY.getText());            
			 if (I <= 0)
				 maxSigmaXY.setText(maxSigmaXYChList.getItem(id).getText()); 		// Update.                    
		 } catch (NumberFormatException e) { 								// If user wrote non numerical test into the field.                
			 maxSigmaXY.setText(maxSigmaXYChList.getItem(id).getText()); 		// Update.
		 }
/*		 // Sigma Z
		 try {
			 int I = Integer.parseInt(minSigmaZ.getText());        		
			 if (I <= 0)
				 minSigmaZ.setText(minSigmaZChList.getItem(id).getText()); 		// Update.                    
		 } catch (NumberFormatException e) { 								// If user wrote non numerical test into the field.                
			 minSigmaZ.setText(minSigmaZChList.getItem(id).getText()); 		// Update.
		 }
		 try {
			 int I = Integer.parseInt(maxSigmaZ.getText());        		
			 if (I <= Integer.parseInt(minSigmaZ.getText()))
				 maxSigmaZ.setText(minSigmaZ.getText());
			 if (I <= 0)
				 maxSigmaZ.setText(maxSigmaZChList.getItem(id).getText()); 		// Update.                    
		 } catch (NumberFormatException e) { 								// If user wrote non numerical test into the field.                
			 maxSigmaZ.setText(maxSigmaZChList.getItem(id).getText()); 		// Update.
		 }        */
		 // Rsquare

		 try {
			 float I = Float.parseFloat(minRsquare.getText());        		
			 if (I < 0)
				 minRsquare.setText(minRsquareChList.getItem(id).getText()); 		// Update.                    
		 } catch (NumberFormatException e) { 								// If user wrote non numerical test into the field.                
			 minRsquare.setText(minRsquareChList.getItem(id).getText()); 		// Update.
		 }
		 try {
			 float I = Float.parseFloat(maxRsquare.getText());        		
			 if (I <= Float.parseFloat(minRsquare.getText()))
				 maxRsquare.setText(minRsquare.getText());
			 if (I <= 0)
				 maxRsquare.setText(maxRsquareChList.getItem(id).getText()); 		// Update.                    
		 } catch (NumberFormatException e) { 								// If user wrote non numerical test into the field.                
			 maxRsquare.setText(maxRsquareChList.getItem(id).getText()); 		// Update.
		 }        
		 // PrecisionXY
		 try {
			 int I = Integer.parseInt(minPrecisionXY.getText());        		
			 if (I < 0)
				 minPrecisionXY.setText(minPrecisionXYChList.getItem(id).getText()); 		// Update.                    
		 } catch (NumberFormatException e) { 								// If user wrote non numerical test into the field.                
			 minPrecisionXY.setText(minPrecisionXYChList.getItem(id).getText()); 		// Update.
		 }
		 try {
			 int I = Integer.parseInt(maxPrecisionXY.getText());        		
			 if (I <= Integer.parseInt(minPrecisionXY.getText()))
				 maxPrecisionXY.setText(minPrecisionXY.getText());
			 if (I <= 0)
				 maxPrecisionXY.setText(maxPrecisionXYChList.getItem(id).getText()); 		// Update.                    
		 } catch (NumberFormatException e) { 								// If user wrote non numerical test into the field.                
			 maxPrecisionXY.setText(maxPrecisionXYChList.getItem(id).getText()); 		// Update.
		 }

		 // PrecisionZ
		 try {
			 int I = Integer.parseInt(minPrecisionZ.getText());        		
			 if (I < 0)
				 minPrecisionZ.setText(minPrecisionZChList.getItem(id).getText()); 		// Update.                    
		 } catch (NumberFormatException e) { 								// If user wrote non numerical test into the field.                
			 minPrecisionZ.setText(minPrecisionZChList.getItem(id).getText()); 		// Update.
		 }
		 try {
			 int I = Integer.parseInt(maxPrecisionZ.getText());        		
			 if (I <= Integer.parseInt(minPrecisionZ.getText()))
				 maxPrecisionZ.setText(minPrecisionZ.getText());
			 if (I <= 0)
				 maxPrecisionZ.setText(maxPrecisionZChList.getItem(id).getText()); 		// Update.                    
		 } catch (NumberFormatException e) { 								// If user wrote non numerical test into the field.                
			 maxPrecisionZ.setText(maxPrecisionZChList.getItem(id).getText()); 		// Update.
		 }

		 // Frame
		 try {
			 int I = Integer.parseInt(minFrame.getText());        		
			 if (I < 0)
				 minFrame.setText(minFrameChList.getItem(id).getText()); 		// Update.                    
		 } catch (NumberFormatException e) { 								// If user wrote non numerical test into the field.                
			 minFrame.setText(minFrameChList.getItem(id).getText()); 		// Update.
		 }
		 try {
			 int I = Integer.parseInt(maxFrame.getText());        		
			 if (I <= Integer.parseInt(minFrame.getText()))
				 maxFrame.setText(minFrame.getText());
			 if (I <= 0)
				 maxFrame.setText(maxFrameChList.getItem(id).getText()); 		// Update.                    
		 } catch (NumberFormatException e) { 								// If user wrote non numerical test into the field.                
			 maxFrame.setText(maxFrameChList.getItem(id).getText()); 		// Update.
		 }           

	 }

	 /*
	  ****************************************************************************
	  ******* Translate user input to arrays of values for calculations **********
	  ****************************************************************************
	  */

	 /*
	  *   get user input pixel size.
	  */
	 private int[] getTotalGain()
	 {        
		 int[] data = new int[10];
		 for (int id = 0; id < data.length; id++)
		 {
			 data[id] = Integer.parseInt(totalGainChList.getItem(id).getText());
		 }
		 return data;
	 }

	 /*
	  *   get minimum pixels above background.
	  */
	 /*	private int[] getMinPixelOverBackground()
	{        
		int[] data = new int[10];
		for (int id = 0; id < data.length; id++)
		{
			data[id] = Integer.parseInt(minPixelOverBackgroundChList.getItem(id).getText());
		}
		return data;
	}
	  */
	 /*
	  *   get minimum pixel intensity for center pixel.
	  */
	 private int[] getMinSignal()
	 {        
		 int[] data = new int[10];
		 for (int id = 0; id < data.length; id++)
		 {
			 data[id] = Integer.parseInt(minimalSignalChList.getItem(id).getText());
		 }
		 return data;
	 }

	 /*
	  *   get window width for background correction.
	  */
	 private int[] getWindowWidth()
	 {        
		 int[] data = new int[10];
		 for (int id = 0; id < data.length; id++)
		 {
			 data[id] = (Integer.parseInt(windowWidthChList.getItem(id).getText())-1)/2;
		 }
		 return data;
	 }

	 /*
	  * render channels.
	  */
	 
	 
	
	 
	 private boolean[] getDoRender()
	 {
		 boolean[] Include = new boolean[10];
		 for (int id = 0; id < 10; id ++)
		 {
			 if(doRenderImageChList.getItem(id).getText().equals("1"))
				 Include[id] = true;
			 else
				 Include[id] = false;
		 }
		 return Include;
	 }

	 /*
	  *   User choice of doing cluster analysis.
	  */
	 private boolean[] getDoClusterAnalysis()
	 {        
		 boolean[] data = new boolean[10];
		 for (int id = 0; id < data.length; id++)
		 {
			 data[id] = doClusterAnalysisChList.getItem(id).isSelected();
		 }
		 return data;
	 }

	 /*
	  *   get epsilon for cluster analysis.
	  */
	 private double[] getEpsilon()
	 {        
		 double[] data = new double[10];
		 for (int id = 0; id < data.length; id++)
		 {
			 data[id] = Float.parseFloat(epsilonChList.getItem(id).getText());
		 }
		 return data;
	 }

	 /*
	  *   get minpts for cluster analysis.
	  */
	 private int[] getMinPtsCluster()
	 {        
		 int[] data = new int[10];
		 for (int id = 0; id < data.length; id++)
		 {
			 data[id] = Integer.parseInt(minPtsClusterChList.getItem(id).getText());
		 }
		 return data;
	 }

	 /*
	  *   get desired output pixel size.
	  */
	 private int[] getOutputPixelSize()
	 {        
		 int[] data = new int[2];
		 /*		for (int id = 0; id < data.length; id++)
		{
			data[id] = Integer.parseInt(outputPixelSizeChList.getItem(id).getText());
		}        */
		 data[0] = Integer.parseInt(outputPixelSize.getText());
		 data[1] = Integer.parseInt(outputPixelSizeZ.getText());
		 return data;
	 }

	 /*
	  *   get minimum particle count for drift correction.
	  */
	 private int[] getDriftCorrBinLowCount()
	 {        
		 int[] data = new int[10];
		 for (int id = 0; id < data.length; id++)
		 {
			 data[id] = Integer.parseInt(driftCorrBinLowCountChList.getItem(id).getText());
		 }
		 return data;
	 }

	 /*
	  *   get maximum particle count for drift correction.
	  */
	 private int[] getDriftCorrBinHighCount()
	 {        
		 int[] data = new int[10];
		 for (int id = 0; id < data.length; id++)
		 {
			 data[id] = Integer.parseInt(driftCorrBinHighCountChList.getItem(id).getText());
		 }
		 return data;
	 }

	 /*
	  *   get maximum shift for drift correction.
	  */

	 private int[][] getDriftCorrShift()
	 {        
		 int[][] data = new int[2][10];
		 for (int id = 0; id < data.length; id++)
		 {
			 data[0][id] = Integer.parseInt(driftCorrShiftXYChList.getItem(id).getText());
			 data[1][id] = Integer.parseInt(driftCorrShiftZChList.getItem(id).getText());
		 }
		 return data;
	 }

	 /*
	  *   get number of bins for drift correction.
	  */
	 private int[] getNumberOfBinsDriftCorr()
	 {        
		 int[] data = new int[10];
		 for (int id = 0; id < data.length; id++)
		 {
			 data[id] = Integer.parseInt(numberOfBinsDriftCorrChList.getItem(id).getText());
		 }
		 return data;
	 }

	 /*
	  *   get minimum particle count for drift correction.
	  */
	 private int[] getChAlignBinLowCount()
	 {         
		 int[] data = new int[10];
		 for (int id = 0; id < data.length; id++)
		 {
			 data[id] = Integer.parseInt(chAlignBinLowCountChList.getItem(id).getText());
		 }
		 return data;
	 }

	 /*
	  *   get maximum particle count for drift correction.
	  */
	 private int[] getChAlignBinHighCount()
	 {        
		 int[] data = new int[10];
		 for (int id = 0; id < data.length; id++)
		 {
			 data[id] = Integer.parseInt(chAlignBinHighCountChList.getItem(id).getText());
		 }
		 return data;
	 }

	 /*
	  *   get maximum shift for channel alignment.
	  */
	 private int[][] getChAlignShift()
	 {        
		 int[][] data = new int[2][10];
		 for (int id = 0; id < data.length; id++)
		 {
			 data[0][id] = Integer.parseInt(chAlignShiftXYChList.getItem(id).getText());
			 data[1][id] = Integer.parseInt(chAlignShiftZChList.getItem(id).getText());
		 }
		 return data;
	 }

	 /*
	  *   get which parameter ranges to use.
	  */

	 private boolean[][] IncludeParameters()
	 {
		 boolean[][] Include = new boolean[7][10];
		 for (int id = 0; id < Include[0].length; id++)
		 {        	
			 if(doPhotonCountChList.getItem(id).getText().equals("1"))
				 Include[0][id] = true;
			 else
				 Include[0][id] = false;
			 if(doSigmaXYChList.getItem(id).getText().equals("1"))
				 Include[1][id] = true;
			 else
				 Include[1][id] = false;
			 if(doSigmaZChList.getItem(id).getText().equals("1"))
				 Include[2][id] = true;
			 else
				 Include[2][id] = false;
			 if(doRsquareChList.getItem(id).getText().equals("1"))
				 Include[3][id] = true;
			 else
				 Include[3][id] = false;
			 if(doPrecisionXYChList.getItem(id).getText().equals("1"))
				 Include[4][id] = true;
			 else
				 Include[4][id] = false;
			 if(doPrecisionZChList.getItem(id).getText().equals("1"))
				 Include[5][id] = true;
			 else
				 Include[5][id] = false;
			 if(doFrameChList.getItem(id).getText().equals("1"))
				 Include[6][id] = true;
			 else
				 Include[6][id] = false;
		 }
		 return Include;
	 }

	 /*
	  *       get lower parameter bounds.
	  */
	 private double[][] lbParameters()
	 {
		 double[][] lb = new double[7][10];
		 for (int id = 0; id < lb[0].length; id++)
		 {
			 lb[0][id] = Float.parseFloat(minPhotonCountChList.getItem(id).getText());
			 lb[1][id] = Float.parseFloat(minSigmaXYChList.getItem(id).getText());
			 //lb[2][id] = Float.parseFloat(minSigmaZChList.getItem(id).getText());
			 lb[2][id] = Float.parseFloat(minRsquareChList.getItem(id).getText());
			 lb[3][id] = Float.parseFloat(minPrecisionXYChList.getItem(id).getText());
			 lb[4][id] = Float.parseFloat(minPrecisionZChList.getItem(id).getText());
			 lb[5][id] = Float.parseFloat(minFrameChList.getItem(id).getText());
		 }
		 return lb;
	 }

	 /*
	  *       get lower parameter bounds.
	  */
	 private double[][] ubParameters()
	 {
		 double[][] ub = new double[7][10];
		 for (int id = 0; id < ub[0].length; id++)
		 {
			 ub[0][id] = Float.parseFloat(maxPhotonCountChList.getItem(id).getText());
			 ub[1][id] = Float.parseFloat(maxSigmaXYChList.getItem(id).getText());
	//		 ub[2][id] = Float.parseFloat(maxSigmaZChList.getItem(id).getText());
			 ub[2][id] = Float.parseFloat(maxRsquareChList.getItem(id).getText());
			 ub[3][id] = Float.parseFloat(maxPrecisionXYChList.getItem(id).getText());
			 ub[4][id] = Float.parseFloat(maxPrecisionZChList.getItem(id).getText());
			 ub[5][id] = Float.parseFloat(maxFrameChList.getItem(id).getText());
		 }
		 return ub;
	 }


	 public void loadParameters(String storeName)
	 {

		 /*
		  * non channel unique variables.
		  */
		 ij.Prefs.set("SMLocalizer.CurrentSetting", storeName); // current.
		 if (ij.Prefs.get("SMLocalizer.settings."+storeName+
				 ".doDriftCorrect.",1) == 1)
			 doDriftCorrect.setSelected(true);		
		 else
			 doDriftCorrect.setSelected(false);
		 if (ij.Prefs.get("SMLocalizer.settings."+storeName+
				 ".doChannelAlign.",1) == 1)
			 doChannelAlign.setSelected(true);		
		 else
			 doChannelAlign.setSelected(false);

		 // pixel size XY
		 outputPixelSize.setText(
				 ij.Prefs.get("SMLocalizer.settings."+storeName+
						 ".pixelSize", 
						 ""));
		 // pixel size Z
		 outputPixelSizeZ.setText(
				 ij.Prefs.get("SMLocalizer.settings."+storeName+
						 ".pixelSizeZ", 
						 ""));
		 // gaussian smoothing
		 if(ij.Prefs.get("SMLocalizer.settings."+storeName+
				 ".doGaussianSmoothing.",1)== 1)
			 doGaussianSmoothing.setSelected(true);
		 else
			 doGaussianSmoothing.setSelected(false);

		 for (int Ch = 0; Ch < 10; Ch++){
			 /*
			  *   Basic input settings
			  */		    		                           		   


			 // total gain
			 totalGainChList.getItem(Ch).setText( 
					 ij.Prefs.get("SMLocalizer.settings."+storeName+
							 ".totaGain."+Ch, 
							 ""));		   
			 // minimum pixel over background
			 /*			minPixelOverBackgroundChList.getItem(Ch).setText( 
					ij.Prefs.get("SMLocalizer.settings."+storeName+
							".minPixelOverBackground."+Ch, 
							""));		   */
			 // minimal signal
			 minimalSignalChList.getItem(Ch).setText( 
					 ij.Prefs.get("SMLocalizer.settings."+storeName+
							 ".minimalSignal."+Ch, 
							 ""));
			 // gauss window size
			 gaussWindowChList.getItem(Ch).setText( 
					 ij.Prefs.get("SMLocalizer.settings."+storeName+
							 ".gaussWindow."+Ch, 
							 ""));
			 // gauss window size
			 windowWidthChList.getItem(Ch).setText( 
					 ij.Prefs.get("SMLocalizer.settings."+storeName+
							 ".windowWidth."+Ch, 
							 ""));

			 /*
			  *       Cluster analysis settings
			  */
			 // min pixel distance
			 doClusterAnalysisChList.getItem(Ch).setText( 
					 ij.Prefs.get("SMLocalizer.settings."+storeName+
							 ".doClusterAnalysis."+Ch, 
							 ""));
			 // min pixel distance			
			 epsilonChList.getItem(Ch).setText( 
					 ij.Prefs.get("SMLocalizer.settings."+storeName+
							 ".epsilon."+Ch, 
							 ""));
			 // min pixel distance
			 minPtsClusterChList.getItem(Ch).setText( 
					 ij.Prefs.get("SMLocalizer.settings."+storeName+
							 ".minPtsCluster."+Ch, 
							 ""));  		    
			 /*
			  *       Render image settings.
			  */
			 // render image
			 doRenderImageChList.getItem(Ch).setText( 
					 ij.Prefs.get("SMLocalizer.settings."+storeName+
							 ".doRenderImage."+Ch, 
							 ""));			

			 /*
			  * store parameter settings:
			  */		    		   					
			 // photon count
			 doPhotonCountChList.getItem(Ch).setText( 
					 ij.Prefs.get("SMLocalizer.settings."+storeName+
							 ".doPotonCount."+Ch, 
							 ""));  
			 minPhotonCountChList.getItem(Ch).setText( 
					 ij.Prefs.get("SMLocalizer.settings."+storeName+
							 ".minPotonCount."+Ch, 
							 ""));  
			 maxPhotonCountChList.getItem(Ch).setText( 
					 ij.Prefs.get("SMLocalizer.settings."+storeName+
							 ".maxPotonCount."+Ch, 
							 ""));  		    
			 // Sigma XY  
			 doSigmaXYChList.getItem(Ch).setText( 
					 ij.Prefs.get("SMLocalizer.settings."+storeName+
							 ".doSigmaXY."+Ch, 
							 ""));  
			 minSigmaXYChList.getItem(Ch).setText( 
					 ij.Prefs.get("SMLocalizer.settings."+storeName+
							 ".minSigmaXY."+Ch, 
							 ""));  
			 maxSigmaXYChList.getItem(Ch).setText( 
					 ij.Prefs.get("SMLocalizer.settings."+storeName+
							 ".maxSigmaXY."+Ch, 
							 ""));  
	/*		 // Sigma Z 		    
			 doSigmaZChList.getItem(Ch).setText( 
					 ij.Prefs.get("SMLocalizer.settings."+storeName+
							 ".doSigmaZ."+Ch, 
							 ""));  
			 minSigmaZChList.getItem(Ch).setText( 
					 ij.Prefs.get("SMLocalizer.settings."+storeName+
							 ".minSigmaZ."+Ch, 
							 ""));  
			 maxSigmaZChList.getItem(Ch).setText( 
					 ij.Prefs.get("SMLocalizer.settings."+storeName+
							 ".maxSigmaZ."+Ch,
							 "")); */
			 // Rsquare  

			 doRsquareChList.getItem(Ch).setText( 
					 ij.Prefs.get("SMLocalizer.settings."+storeName+
							 ".doRsquare."+Ch, 
							 ""));  
			 minRsquareChList.getItem(Ch).setText( 
					 ij.Prefs.get("SMLocalizer.settings."+storeName+
							 ".minRsquare."+Ch, 
							 ""));  
			 maxRsquareChList.getItem(Ch).setText( 
					 ij.Prefs.get("SMLocalizer.settings."+storeName+
							 ".maxRsquare."+Ch,
							 ""));		    		    

			 // Precision XY
			 doPrecisionXYChList.getItem(Ch).setText( 
					 ij.Prefs.get("SMLocalizer.settings."+storeName+
							 ".doPrecisionXY."+Ch, 
							 ""));  
			 minPrecisionXYChList.getItem(Ch).setText( 
					 ij.Prefs.get("SMLocalizer.settings."+storeName+
							 ".minPrecisionXY."+Ch, 
							 ""));  
			 maxPrecisionXYChList.getItem(Ch).setText( 
					 ij.Prefs.get("SMLocalizer.settings."+storeName+
							 ".maxPrecisionXY."+Ch,
							 ""));		   
			 // Precision Z
			 doPrecisionZChList.getItem(Ch).setText( 
					 ij.Prefs.get("SMLocalizer.settings."+storeName+
							 ".doPrecisionZ."+Ch, 
							 ""));  
			 minPrecisionZChList.getItem(Ch).setText( 
					 ij.Prefs.get("SMLocalizer.settings."+storeName+
							 ".minPrecisionZ."+Ch, 
							 ""));  
			 maxPrecisionZChList.getItem(Ch).setText( 
					 ij.Prefs.get("SMLocalizer.settings."+storeName+
							 ".maxPrecisionZ."+Ch,
							 ""));		    
			 // Frame
			 doFrameChList.getItem(Ch).setText( 
					 ij.Prefs.get("SMLocalizer.settings."+storeName+
							 ".doFrame."+Ch, 
							 ""));  
			 minFrameChList.getItem(Ch).setText( 
					 ij.Prefs.get("SMLocalizer.settings."+storeName+
							 ".minFrame."+Ch, 
							 ""));  
			 maxFrameChList.getItem(Ch).setText( 
					 ij.Prefs.get("SMLocalizer.settings."+storeName+
							 ".maxFrame."+Ch,
							 ""));		
			 /*
			  *   Drift and channel correct settings
			  */

			 // drift correction bins.
			 driftCorrBinLowCountChList.getItem(Ch).setText( 
					 ij.Prefs.get("SMLocalizer.settings."+storeName+
							 ".driftCorrBinLow."+Ch, 
							 ""));  
			 driftCorrBinHighCountChList.getItem(Ch).setText( 
					 ij.Prefs.get("SMLocalizer.settings."+storeName+
							 ".driftCorrBinHigh."+Ch,
							 ""));

			 // drift correction shift
			 driftCorrShiftXYChList.getItem(Ch).setText( 
					 ij.Prefs.get("SMLocalizer.settings."+storeName+
							 ".driftCorrShiftXY."+Ch, 
							 ""));  
			 driftCorrShiftZChList.getItem(Ch).setText( 
					 ij.Prefs.get("SMLocalizer.settings."+storeName+
							 ".driftCorrShiftZ."+Ch,
							 ""));			    
			 // number of drift bins
			 numberOfBinsDriftCorrChList.getItem(Ch).setText( 
					 ij.Prefs.get("SMLocalizer.settings."+storeName+
							 ".driftCorrBin."+Ch,
							 ""));	


			 // channel align bin low
			 chAlignBinLowCountChList.getItem(Ch).setText( 
					 ij.Prefs.get("SMLocalizer.settings."+storeName+
							 ".chAlignBinLow."+Ch,
							 ""));	

			 // channel align bin high
			 chAlignBinHighCountChList.getItem(Ch).setText( 
					 ij.Prefs.get("SMLocalizer.settings."+storeName+
							 ".chAlignBinHigh."+Ch,
							 ""));	


			 // channel align shift
			 chAlignShiftXYChList.getItem(Ch).setText( 
					 ij.Prefs.get("SMLocalizer.settings."+storeName+
							 ".chAlignShiftXY."+Ch,
							 ""));	
			 chAlignShiftZChList.getItem(Ch).setText( 
					 ij.Prefs.get("SMLocalizer.settings."+storeName+
							 ".chAlignShiftZ."+Ch,
							 ""));	 
			 doCorrelativeChList.getItem(Ch).setText( 
					 ij.Prefs.get("SMLocalizer.settings."+storeName+
							 ".correlative."+Ch,
							 ""));
			 doChromaticChList.getItem(Ch).setText( 
					 ij.Prefs.get("SMLocalizer.settings."+storeName+
							 ".chromatic."+Ch,
							 ""));				
			 fiducialsChList.getItem(Ch).setText( 
					 ij.Prefs.get("SMLocalizer.settings."+storeName+
							 ".fiducials."+Ch,
							 ""));	
			 
			 
		 }

		 int id = 0; // set current ch to 1.
		 updateVisible(id); // update fields that user can see.

	 } // loadParameters


	 public void setParameters(String storeName)
	 {
		 updateList(channelId.getSelectedIndex()-1);  // verify that current channel have ok set fields:

		 // pop list of current settings, first entry = add new.

		 /*
		  * non channel unique variables.
		  */

		 ij.Prefs.set("SMLocalizer.CurrentSetting",storeName);


		 if (doDriftCorrect.isSelected())
			 ij.Prefs.set("SMLocalizer.settings."+storeName+
					 ".doDriftCorrect.",1);
		 else
			 ij.Prefs.set("SMLocalizer.settings."+storeName+
					 ".doDriftCorrect.",0);

		 if (doChannelAlign.isSelected())
			 ij.Prefs.set("SMLocalizer.settings."+storeName+
					 ".doChannelAlign.",1);
		 else
			 ij.Prefs.set("SMLocalizer.settings."+storeName+
					 ".doChannelAlign.",0);		

		 // pixel size XY
		 ij.Prefs.set("SMLocalizer.settings."+storeName+
				 ".pixelSize", 
				 outputPixelSize.getText());
		 // pixel size Z
		 ij.Prefs.set("SMLocalizer.settings."+storeName+
				 ".pixelSizeZ", 
				 outputPixelSizeZ.getText());
		 // gaussian smoothing
		 if(doGaussianSmoothing.isSelected())
			 ij.Prefs.set("SMLocalizer.settings."+storeName+
					 ".doGaussianSmoothing.",1);
		 else
			 ij.Prefs.set("SMLocalizer.settings."+storeName+
					 ".doGaussianSmoothing.",0);

		 for (int Ch = 0; Ch < 10; Ch++){
			 /*
			  *   Basic input settings
			  */		    		                           		   

			 // total gain
			 ij.Prefs.set("SMLocalizer.settings."+storeName+
					 ".totaGain."+Ch, 
					 totalGainChList.getItem(Ch).getText());
			 // minimum pixel over background
			 /*			ij.Prefs.set("SMLocalizer.settings."+storeName+
					".minPixelOverBackground."+Ch, 
					minPixelOverBackgroundChList.getItem(Ch).getText());*/
			 // minimal signal
			 ij.Prefs.set("SMLocalizer.settings."+storeName+
					 ".minimalSignal."+Ch, 
					 minimalSignalChList.getItem(Ch).getText());
			 // gauss window size
			 ij.Prefs.set("SMLocalizer.settings."+storeName+
					 ".gaussWindow."+Ch, 
					 gaussWindowChList.getItem(Ch).getText());
			 // gauss window size
			 ij.Prefs.set("SMLocalizer.settings."+storeName+
					 ".windowWidth."+Ch, 
					 windowWidthChList.getItem(Ch).getText());



			 /*
			  *       Cluster analysis settings
			  */
			 // min pixel distance
			 ij.Prefs.set("SMLocalizer.settings."+storeName+
					 ".doClusterAnalysis."+Ch, 
					 doClusterAnalysisChList.getItem(Ch).getText());
			 // min pixel distance
			 ij.Prefs.set("SMLocalizer.settings."+storeName+
					 ".epsilon."+Ch, 
					 epsilonChList.getItem(Ch).getText());
			 // min pixel distance
			 ij.Prefs.set("SMLocalizer.settings."+storeName+
					 ".minPtsCluster."+Ch, 
					 minPtsClusterChList.getItem(Ch).getText());		    


			 /*
			  *       Render image settings.
			  */
			 // render image		
			 ij.Prefs.set("SMLocalizer.settings."+storeName+
					 ".doRenderImage."+Ch,
					 doRenderImageChList.getItem(Ch).getText());		


			 /*
			  * store parameter settings:
			  */		    		   					
			 // photon count
			 ij.Prefs.set("SMLocalizer.settings."+storeName+
					 ".doPotonCount."+Ch, 
					 doPhotonCountChList.getItem(Ch).getText());
			 ij.Prefs.set("SMLocalizer.settings."+storeName+
					 ".minPotonCount."+Ch, 
					 minPhotonCountChList.getItem(Ch).getText());
			 ij.Prefs.set("SMLocalizer.settings."+storeName+
					 ".maxPotonCount."+Ch, 
					 maxPhotonCountChList.getItem(Ch).getText());

			 // Sigma XY        		    
			 ij.Prefs.set("SMLocalizer.settings."+storeName+
					 ".doSigmaXY."+Ch, 
					 doSigmaXYChList.getItem(Ch).getText());
			 ij.Prefs.set("SMLocalizer.settings."+storeName+
					 ".minSigmaXY."+Ch, 
					 minSigmaXYChList.getItem(Ch).getText());
			 ij.Prefs.set("SMLocalizer.settings."+storeName+
					 ".maxSigmaXY."+Ch, 
					 maxSigmaXYChList.getItem(Ch).getText());

/*			 // Sigma Z        		    
			 ij.Prefs.set("SMLocalizer.settings."+storeName+
					 ".doSigmaZ."+Ch, 
					 doSigmaZChList.getItem(Ch).getText());
			 ij.Prefs.set("SMLocalizer.settings."+storeName+
					 ".minSigmaZ."+Ch, 
					 minSigmaZChList.getItem(Ch).getText());
			 ij.Prefs.set("SMLocalizer.settings."+storeName+
					 ".maxSigmaZ."+Ch, 
					 maxSigmaZChList.getItem(Ch).getText());
*/
			 // Rsquare        
			 ij.Prefs.set("SMLocalizer.settings."+storeName+
					 ".doRsquare."+Ch, 
					 doRsquareChList.getItem(Ch).getText());
			 ij.Prefs.set("SMLocalizer.settings."+storeName+
					 ".minRsquare."+Ch, 
					 minRsquareChList.getItem(Ch).getText());
			 ij.Prefs.set("SMLocalizer.settings."+storeName+
					 ".maxRsquare."+Ch, 
					 maxRsquareChList.getItem(Ch).getText());

			 // Precision XY
			 ij.Prefs.set("SMLocalizer.settings."+storeName+
					 ".doPrecisionXY."+Ch, 
					 doPrecisionXYChList.getItem(Ch).getText());
			 ij.Prefs.set("SMLocalizer.settings."+storeName+
					 ".minPrecisionXY."+Ch, 
					 minPrecisionXYChList.getItem(Ch).getText());
			 ij.Prefs.set("SMLocalizer.settings."+storeName+
					 ".maxPrecisionXY."+Ch, 
					 maxPrecisionXYChList.getItem(Ch).getText());

			 // Precision Z
			 ij.Prefs.set("SMLocalizer.settings."+storeName+
					 ".doPrecisionZ."+Ch, 
					 doPrecisionZChList.getItem(Ch).getText());
			 ij.Prefs.set("SMLocalizer.settings."+storeName+
					 ".minPrecisionZ."+Ch, 
					 minPrecisionZChList.getItem(Ch).getText());
			 ij.Prefs.set("SMLocalizer.settings."+storeName+
					 ".maxPrecisionZ."+Ch, 
					 maxPrecisionZChList.getItem(Ch).getText());
			 // Frame
			 ij.Prefs.set("SMLocalizer.settings."+storeName+
					 ".doFrame."+Ch, 
					 doFrameChList.getItem(Ch).getText());
			 ij.Prefs.set("SMLocalizer.settings."+storeName+
					 ".minFrame."+Ch, 
					 minFrameChList.getItem(Ch).getText());
			 ij.Prefs.set("SMLocalizer.settings."+storeName+
					 ".maxFrame."+Ch, 
					 maxFrameChList.getItem(Ch).getText());
			 /*
			  *   Drift and channel correct settings
			  */

			 // drift correction bins.
			 ij.Prefs.set("SMLocalizer.settings."+storeName+
					 ".driftCorrBinLow."+Ch, 
					 driftCorrBinLowCountChList.getItem(Ch).getText());
			 ij.Prefs.set("SMLocalizer.settings."+storeName+
					 ".driftCorrBinHigh."+Ch, 
					 driftCorrBinHighCountChList.getItem(Ch).getText());

			 // drift correction shift
			 ij.Prefs.set("SMLocalizer.settings."+storeName+
					 ".driftCorrShiftXY."+Ch, 
					 driftCorrShiftXYChList.getItem(Ch).getText());
			 ij.Prefs.set("SMLocalizer.settings."+storeName+
					 ".driftCorrShiftZ."+Ch, 
					 driftCorrShiftZChList.getItem(Ch).getText());

			 // number of drift bins
			 ij.Prefs.set("SMLocalizer.settings."+storeName+
					 ".driftCorrBin."+Ch, 
					 numberOfBinsDriftCorrChList.getItem(Ch).getText());

			 // channel align bin low
			 ij.Prefs.set("SMLocalizer.settings."+storeName+
					 ".chAlignBinLow."+Ch, 
					 chAlignBinLowCountChList.getItem(Ch).getText());
			 // channel align bin high
			 ij.Prefs.set("SMLocalizer.settings."+storeName+
					 ".chAlignBinHigh."+Ch, 
					 chAlignBinHighCountChList.getItem(Ch).getText());

			 // channel align shift
			 ij.Prefs.set("SMLocalizer.settings."+storeName+
					 ".chAlignShiftXY."+Ch, 
					 chAlignShiftXYChList.getItem(Ch).getText());
			 ij.Prefs.set("SMLocalizer.settings."+storeName+
					 ".chAlignShiftZ."+Ch, 
					 chAlignShiftZChList.getItem(Ch).getText());	
			 
			 ij.Prefs.set("SMLocalizer.settings."+storeName+
					 ".correlative."+Ch, 
					 doCorrelativeChList.getItem(Ch).getText());
			 ij.Prefs.set("SMLocalizer.settings."+storeName+
					 ".chromatic."+Ch, 
					 doChromaticChList.getItem(Ch).getText());
			 ij.Prefs.set("SMLocalizer.settings."+storeName+
					 ".fiducials."+Ch, 
					 fiducialsChList.getItem(Ch).getText());	 			 
		 }

		 ij.Prefs.savePreferences(); // store settings. 
	 } // setParameters
	 /**
	  * @param args the command line arguments
	  */
	 public static void main(String args[]) {
		 /* Set the Nimbus look and feel */
		 //<editor-fold defaultstate="collapsed" desc=" Look and feel setting code (optional) ">
		 /* If Nimbus (introduced in Java SE 6) is not available, stay with the default look and feel.
		  * For details see http://download.oracle.com/javase/tutorial/uiswing/lookandfeel/plaf.html 
		  */
		 try {
			 for (javax.swing.UIManager.LookAndFeelInfo info : javax.swing.UIManager.getInstalledLookAndFeels()) {
				 if ("Nimbus".equals(info.getName())) {
					 javax.swing.UIManager.setLookAndFeel(info.getClassName());
					 break;
				 }
			 }
		 } catch (ClassNotFoundException ex) {
			 java.util.logging.Logger.getLogger(SMLocalizerGUI.class.getName()).log(java.util.logging.Level.SEVERE, null, ex);
		 } catch (InstantiationException ex) {
			 java.util.logging.Logger.getLogger(SMLocalizerGUI.class.getName()).log(java.util.logging.Level.SEVERE, null, ex);
		 } catch (IllegalAccessException ex) {
			 java.util.logging.Logger.getLogger(SMLocalizerGUI.class.getName()).log(java.util.logging.Level.SEVERE, null, ex);
		 } catch (javax.swing.UnsupportedLookAndFeelException ex) {
			 java.util.logging.Logger.getLogger(SMLocalizerGUI.class.getName()).log(java.util.logging.Level.SEVERE, null, ex);
		 }
		 //</editor-fold>
		 //</editor-fold>

		 /* Create and display the form */
		 java.awt.EventQueue.invokeLater(new Runnable() {
			 public void run() {
				 new SMLocalizerGUI().setVisible(true);
			 }
		 });
	 }
	 // Variables declaration - do not modify                     
	    private javax.swing.JPanel Analysis;
	    private javax.swing.JPanel BasicInp;
	    private javax.swing.JRadioButton GPUcomputation;
	    private javax.swing.JLabel Header;
	    private javax.swing.JLabel ParameterLabel;
	    private javax.swing.JPanel ParameterRange;
	    private javax.swing.JButton Process;
	    private javax.swing.JMenuItem ROIsizeData1;
	    private javax.swing.JMenuItem ROIsizeData10;
	    private javax.swing.JMenuItem ROIsizeData2;
	    private javax.swing.JMenuItem ROIsizeData3;
	    private javax.swing.JMenuItem ROIsizeData4;
	    private javax.swing.JMenuItem ROIsizeData5;
	    private javax.swing.JMenuItem ROIsizeData6;
	    private javax.swing.JMenuItem ROIsizeData7;
	    private javax.swing.JMenuItem ROIsizeData8;
	    private javax.swing.JMenuItem ROIsizeData9;
	    private javax.swing.JLabel XYrenderLabel;
	    private javax.swing.JLabel ZrenderLabel;
	    private javax.swing.JButton alignChannels;
	    private javax.swing.JLabel basicInput;
	    private javax.swing.ButtonGroup buttonGroup2;
	    private javax.swing.JButton calibrate;
	    private javax.swing.JTextField chAlignBinHighCount;
	    private javax.swing.JMenu chAlignBinHighCountChList;
	    private javax.swing.JMenuItem chAlignBinHighCountData1;
	    private javax.swing.JMenuItem chAlignBinHighCountData10;
	    private javax.swing.JMenuItem chAlignBinHighCountData2;
	    private javax.swing.JMenuItem chAlignBinHighCountData3;
	    private javax.swing.JMenuItem chAlignBinHighCountData4;
	    private javax.swing.JMenuItem chAlignBinHighCountData5;
	    private javax.swing.JMenuItem chAlignBinHighCountData6;
	    private javax.swing.JMenuItem chAlignBinHighCountData7;
	    private javax.swing.JMenuItem chAlignBinHighCountData8;
	    private javax.swing.JMenuItem chAlignBinHighCountData9;
	    private javax.swing.JTextField chAlignBinLowCount;
	    private javax.swing.JMenu chAlignBinLowCountChList;
	    private javax.swing.JMenuItem chAlignBinLowCountData1;
	    private javax.swing.JMenuItem chAlignBinLowCountData10;
	    private javax.swing.JMenuItem chAlignBinLowCountData2;
	    private javax.swing.JMenuItem chAlignBinLowCountData3;
	    private javax.swing.JMenuItem chAlignBinLowCountData4;
	    private javax.swing.JMenuItem chAlignBinLowCountData5;
	    private javax.swing.JMenuItem chAlignBinLowCountData6;
	    private javax.swing.JMenuItem chAlignBinLowCountData7;
	    private javax.swing.JMenuItem chAlignBinLowCountData8;
	    private javax.swing.JMenuItem chAlignBinLowCountData9;
	    private javax.swing.JTextField chAlignShiftXY;
	    private javax.swing.JMenu chAlignShiftXYChList;
	    private javax.swing.JMenuItem chAlignShiftXYData1;
	    private javax.swing.JMenuItem chAlignShiftXYData10;
	    private javax.swing.JMenuItem chAlignShiftXYData2;
	    private javax.swing.JMenuItem chAlignShiftXYData3;
	    private javax.swing.JMenuItem chAlignShiftXYData4;
	    private javax.swing.JMenuItem chAlignShiftXYData5;
	    private javax.swing.JMenuItem chAlignShiftXYData6;
	    private javax.swing.JMenuItem chAlignShiftXYData7;
	    private javax.swing.JMenuItem chAlignShiftXYData8;
	    private javax.swing.JMenuItem chAlignShiftXYData9;
	    private javax.swing.JTextField chAlignShiftZ;
	    private javax.swing.JMenu chAlignShiftZChList;
	    private javax.swing.JMenuItem chAlignShiftZData1;
	    private javax.swing.JMenuItem chAlignShiftZData10;
	    private javax.swing.JMenuItem chAlignShiftZData2;
	    private javax.swing.JMenuItem chAlignShiftZData3;
	    private javax.swing.JMenuItem chAlignShiftZData4;
	    private javax.swing.JMenuItem chAlignShiftZData5;
	    private javax.swing.JMenuItem chAlignShiftZData6;
	    private javax.swing.JMenuItem chAlignShiftZData7;
	    private javax.swing.JMenuItem chAlignShiftZData8;
	    private javax.swing.JMenuItem chAlignShiftZData9;
	    private javax.swing.JComboBox<String> channelId;
	    private javax.swing.JButton cleanTable;
	    private javax.swing.JButton clusterAnalysis;
	    private javax.swing.JButton correctBackground;
	    private javax.swing.JCheckBox doChannelAlign;
	    private javax.swing.JMenu doChromaticChList;
	    private javax.swing.JMenuItem doChromaticData1;
	    private javax.swing.JMenuItem doChromaticData10;
	    private javax.swing.JMenuItem doChromaticData2;
	    private javax.swing.JMenuItem doChromaticData3;
	    private javax.swing.JMenuItem doChromaticData4;
	    private javax.swing.JMenuItem doChromaticData5;
	    private javax.swing.JMenuItem doChromaticData6;
	    private javax.swing.JMenuItem doChromaticData7;
	    private javax.swing.JMenuItem doChromaticData8;
	    private javax.swing.JMenuItem doChromaticData9;
	    private javax.swing.JCheckBox doClusterAnalysis;
	    private javax.swing.JMenu doClusterAnalysisChList;
	    private javax.swing.JMenuItem doClusterAnalysisData1;
	    private javax.swing.JMenuItem doClusterAnalysisData10;
	    private javax.swing.JMenuItem doClusterAnalysisData2;
	    private javax.swing.JMenuItem doClusterAnalysisData3;
	    private javax.swing.JMenuItem doClusterAnalysisData4;
	    private javax.swing.JMenuItem doClusterAnalysisData5;
	    private javax.swing.JMenuItem doClusterAnalysisData6;
	    private javax.swing.JMenuItem doClusterAnalysisData7;
	    private javax.swing.JMenuItem doClusterAnalysisData8;
	    private javax.swing.JMenuItem doClusterAnalysisData9;
	    private javax.swing.JMenu doCorrelativeChList;
	    private javax.swing.JMenuItem doCorrelativeData1;
	    private javax.swing.JMenuItem doCorrelativeData10;
	    private javax.swing.JMenuItem doCorrelativeData2;
	    private javax.swing.JMenuItem doCorrelativeData3;
	    private javax.swing.JMenuItem doCorrelativeData4;
	    private javax.swing.JMenuItem doCorrelativeData5;
	    private javax.swing.JMenuItem doCorrelativeData6;
	    private javax.swing.JMenuItem doCorrelativeData7;
	    private javax.swing.JMenuItem doCorrelativeData8;
	    private javax.swing.JMenuItem doCorrelativeData9;
	    private javax.swing.JCheckBox doDriftCorrect;
	    private javax.swing.JCheckBox doFrame;
	    private javax.swing.JMenu doFrameChList;
	    private javax.swing.JMenuItem doFrameData1;
	    private javax.swing.JMenuItem doFrameData10;
	    private javax.swing.JMenuItem doFrameData2;
	    private javax.swing.JMenuItem doFrameData3;
	    private javax.swing.JMenuItem doFrameData4;
	    private javax.swing.JMenuItem doFrameData5;
	    private javax.swing.JMenuItem doFrameData6;
	    private javax.swing.JMenuItem doFrameData7;
	    private javax.swing.JMenuItem doFrameData8;
	    private javax.swing.JMenuItem doFrameData9;
	    private javax.swing.JCheckBox doGaussianSmoothing;
	    private javax.swing.JCheckBox doPhotonCount;
	    private javax.swing.JMenu doPhotonCountChList;
	    private javax.swing.JMenuItem doPhotonCountData1;
	    private javax.swing.JMenuItem doPhotonCountData10;
	    private javax.swing.JMenuItem doPhotonCountData2;
	    private javax.swing.JMenuItem doPhotonCountData3;
	    private javax.swing.JMenuItem doPhotonCountData4;
	    private javax.swing.JMenuItem doPhotonCountData5;
	    private javax.swing.JMenuItem doPhotonCountData6;
	    private javax.swing.JMenuItem doPhotonCountData7;
	    private javax.swing.JMenuItem doPhotonCountData8;
	    private javax.swing.JMenuItem doPhotonCountData9;
	    private javax.swing.JCheckBox doPrecisionXY;
	    private javax.swing.JMenu doPrecisionXYChList;
	    private javax.swing.JMenuItem doPrecisionXYData1;
	    private javax.swing.JMenuItem doPrecisionXYData10;
	    private javax.swing.JMenuItem doPrecisionXYData2;
	    private javax.swing.JMenuItem doPrecisionXYData3;
	    private javax.swing.JMenuItem doPrecisionXYData4;
	    private javax.swing.JMenuItem doPrecisionXYData5;
	    private javax.swing.JMenuItem doPrecisionXYData6;
	    private javax.swing.JMenuItem doPrecisionXYData7;
	    private javax.swing.JMenuItem doPrecisionXYData8;
	    private javax.swing.JMenuItem doPrecisionXYData9;
	    private javax.swing.JCheckBox doPrecisionZ;
	    private javax.swing.JMenu doPrecisionZChList;
	    private javax.swing.JMenuItem doPrecisionZData1;
	    private javax.swing.JMenuItem doPrecisionZData10;
	    private javax.swing.JMenuItem doPrecisionZData2;
	    private javax.swing.JMenuItem doPrecisionZData3;
	    private javax.swing.JMenuItem doPrecisionZData4;
	    private javax.swing.JMenuItem doPrecisionZData5;
	    private javax.swing.JMenuItem doPrecisionZData6;
	    private javax.swing.JMenuItem doPrecisionZData7;
	    private javax.swing.JMenuItem doPrecisionZData8;
	    private javax.swing.JMenuItem doPrecisionZData9;
	    private javax.swing.JCheckBox doRenderImage;
	    private javax.swing.JMenu doRenderImageChList;
	    private javax.swing.JMenuItem doRenderImageData1;
	    private javax.swing.JMenuItem doRenderImageData10;
	    private javax.swing.JMenuItem doRenderImageData2;
	    private javax.swing.JMenuItem doRenderImageData3;
	    private javax.swing.JMenuItem doRenderImageData4;
	    private javax.swing.JMenuItem doRenderImageData5;
	    private javax.swing.JMenuItem doRenderImageData6;
	    private javax.swing.JMenuItem doRenderImageData7;
	    private javax.swing.JMenuItem doRenderImageData8;
	    private javax.swing.JMenuItem doRenderImageData9;
	    private javax.swing.JCheckBox doRsquare;
	    private javax.swing.JMenu doRsquareChList;
	    private javax.swing.JMenuItem doRsquareData1;
	    private javax.swing.JMenuItem doRsquareData10;
	    private javax.swing.JMenuItem doRsquareData2;
	    private javax.swing.JMenuItem doRsquareData3;
	    private javax.swing.JMenuItem doRsquareData4;
	    private javax.swing.JMenuItem doRsquareData5;
	    private javax.swing.JMenuItem doRsquareData6;
	    private javax.swing.JMenuItem doRsquareData7;
	    private javax.swing.JMenuItem doRsquareData8;
	    private javax.swing.JMenuItem doRsquareData9;
	    private javax.swing.JCheckBox doSigmaXY;
	    private javax.swing.JMenu doSigmaXYChList;
	    private javax.swing.JMenuItem doSigmaXYData1;
	    private javax.swing.JMenuItem doSigmaXYData10;
	    private javax.swing.JMenuItem doSigmaXYData2;
	    private javax.swing.JMenuItem doSigmaXYData3;
	    private javax.swing.JMenuItem doSigmaXYData4;
	    private javax.swing.JMenuItem doSigmaXYData5;
	    private javax.swing.JMenuItem doSigmaXYData6;
	    private javax.swing.JMenuItem doSigmaXYData7;
	    private javax.swing.JMenuItem doSigmaXYData8;
	    private javax.swing.JMenuItem doSigmaXYData9;
	    private javax.swing.JMenu doSigmaZChList;
	    private javax.swing.JMenuItem doSigmaZData1;
	    private javax.swing.JMenuItem doSigmaZData10;
	    private javax.swing.JMenuItem doSigmaZData2;
	    private javax.swing.JMenuItem doSigmaZData3;
	    private javax.swing.JMenuItem doSigmaZData4;
	    private javax.swing.JMenuItem doSigmaZData5;
	    private javax.swing.JMenuItem doSigmaZData6;
	    private javax.swing.JMenuItem doSigmaZData7;
	    private javax.swing.JMenuItem doSigmaZData8;
	    private javax.swing.JMenuItem doSigmaZData9;
	    private javax.swing.JTextField driftCorrBinHighCount;
	    private javax.swing.JMenu driftCorrBinHighCountChList;
	    private javax.swing.JMenuItem driftCorrBinHighCountData1;
	    private javax.swing.JMenuItem driftCorrBinHighCountData10;
	    private javax.swing.JMenuItem driftCorrBinHighCountData2;
	    private javax.swing.JMenuItem driftCorrBinHighCountData3;
	    private javax.swing.JMenuItem driftCorrBinHighCountData4;
	    private javax.swing.JMenuItem driftCorrBinHighCountData5;
	    private javax.swing.JMenuItem driftCorrBinHighCountData6;
	    private javax.swing.JMenuItem driftCorrBinHighCountData7;
	    private javax.swing.JMenuItem driftCorrBinHighCountData8;
	    private javax.swing.JMenuItem driftCorrBinHighCountData9;
	    private javax.swing.JTextField driftCorrBinLowCount;
	    private javax.swing.JMenu driftCorrBinLowCountChList;
	    private javax.swing.JMenuItem driftCorrBinLowCountData1;
	    private javax.swing.JMenuItem driftCorrBinLowCountData10;
	    private javax.swing.JMenuItem driftCorrBinLowCountData2;
	    private javax.swing.JMenuItem driftCorrBinLowCountData3;
	    private javax.swing.JMenuItem driftCorrBinLowCountData4;
	    private javax.swing.JMenuItem driftCorrBinLowCountData5;
	    private javax.swing.JMenuItem driftCorrBinLowCountData6;
	    private javax.swing.JMenuItem driftCorrBinLowCountData7;
	    private javax.swing.JMenuItem driftCorrBinLowCountData8;
	    private javax.swing.JMenuItem driftCorrBinLowCountData9;
	    private javax.swing.JTextField driftCorrShiftXY;
	    private javax.swing.JMenu driftCorrShiftXYChList;
	    private javax.swing.JMenuItem driftCorrShiftXYData1;
	    private javax.swing.JMenuItem driftCorrShiftXYData10;
	    private javax.swing.JMenuItem driftCorrShiftXYData2;
	    private javax.swing.JMenuItem driftCorrShiftXYData3;
	    private javax.swing.JMenuItem driftCorrShiftXYData4;
	    private javax.swing.JMenuItem driftCorrShiftXYData5;
	    private javax.swing.JMenuItem driftCorrShiftXYData6;
	    private javax.swing.JMenuItem driftCorrShiftXYData7;
	    private javax.swing.JMenuItem driftCorrShiftXYData8;
	    private javax.swing.JMenuItem driftCorrShiftXYData9;
	    private javax.swing.JTextField driftCorrShiftZ;
	    private javax.swing.JMenu driftCorrShiftZChList;
	    private javax.swing.JMenuItem driftCorrShiftZData1;
	    private javax.swing.JMenuItem driftCorrShiftZData10;
	    private javax.swing.JMenuItem driftCorrShiftZData2;
	    private javax.swing.JMenuItem driftCorrShiftZData3;
	    private javax.swing.JMenuItem driftCorrShiftZData4;
	    private javax.swing.JMenuItem driftCorrShiftZData5;
	    private javax.swing.JMenuItem driftCorrShiftZData6;
	    private javax.swing.JMenuItem driftCorrShiftZData7;
	    private javax.swing.JMenuItem driftCorrShiftZData8;
	    private javax.swing.JMenuItem driftCorrShiftZData9;
	    private javax.swing.JButton driftCorrect;
	    private javax.swing.JTextField epsilon;
	    private javax.swing.JMenu epsilonChList;
	    private javax.swing.JMenuItem epsilonData1;
	    private javax.swing.JMenuItem epsilonData10;
	    private javax.swing.JMenuItem epsilonData2;
	    private javax.swing.JMenuItem epsilonData3;
	    private javax.swing.JMenuItem epsilonData4;
	    private javax.swing.JMenuItem epsilonData5;
	    private javax.swing.JMenuItem epsilonData6;
	    private javax.swing.JMenuItem epsilonData7;
	    private javax.swing.JMenuItem epsilonData8;
	    private javax.swing.JMenuItem epsilonData9;
	    private javax.swing.JLabel epsilonLabel;
	    private javax.swing.JMenu fiducialsChList;
	    private javax.swing.JMenuItem fiducialsChoice1;
	    private javax.swing.JMenuItem fiducialsChoice10;
	    private javax.swing.JMenuItem fiducialsChoice2;
	    private javax.swing.JMenuItem fiducialsChoice3;
	    private javax.swing.JMenuItem fiducialsChoice4;
	    private javax.swing.JMenuItem fiducialsChoice5;
	    private javax.swing.JMenuItem fiducialsChoice6;
	    private javax.swing.JMenuItem fiducialsChoice7;
	    private javax.swing.JMenuItem fiducialsChoice8;
	    private javax.swing.JMenuItem fiducialsChoice9;
	    private javax.swing.JMenu gaussWindowChList;
	    private javax.swing.JTextField inputPixelSize;
	    private javax.swing.JLabel inputPixelSizeLabel;
	    private javax.swing.JPanel jPanel1;
	    private javax.swing.JPanel jPanel2;
	    private javax.swing.JPanel jPanel3;
	    private javax.swing.JPanel jPanel4;
	    private javax.swing.JButton loadSettings;
	    private javax.swing.JButton localize_Fit;
	    private javax.swing.JTextField maxFrame;
	    private javax.swing.JMenu maxFrameChList;
	    private javax.swing.JMenuItem maxFrameData1;
	    private javax.swing.JMenuItem maxFrameData10;
	    private javax.swing.JMenuItem maxFrameData2;
	    private javax.swing.JMenuItem maxFrameData3;
	    private javax.swing.JMenuItem maxFrameData4;
	    private javax.swing.JMenuItem maxFrameData5;
	    private javax.swing.JMenuItem maxFrameData6;
	    private javax.swing.JMenuItem maxFrameData7;
	    private javax.swing.JMenuItem maxFrameData8;
	    private javax.swing.JMenuItem maxFrameData9;
	    private javax.swing.JLabel maxLabel;
	    private javax.swing.JLabel maxLabel1;
	    private javax.swing.JLabel maxLabel2;
	    private javax.swing.JLabel maxLabel3;
	    private javax.swing.JLabel maxLabel4;
	    private javax.swing.JTextField maxPhotonCount;
	    private javax.swing.JMenu maxPhotonCountChList;
	    private javax.swing.JMenuItem maxPhotonCountData1;
	    private javax.swing.JMenuItem maxPhotonCountData10;
	    private javax.swing.JMenuItem maxPhotonCountData2;
	    private javax.swing.JMenuItem maxPhotonCountData3;
	    private javax.swing.JMenuItem maxPhotonCountData4;
	    private javax.swing.JMenuItem maxPhotonCountData5;
	    private javax.swing.JMenuItem maxPhotonCountData6;
	    private javax.swing.JMenuItem maxPhotonCountData7;
	    private javax.swing.JMenuItem maxPhotonCountData8;
	    private javax.swing.JMenuItem maxPhotonCountData9;
	    private javax.swing.JTextField maxPrecisionXY;
	    private javax.swing.JMenu maxPrecisionXYChList;
	    private javax.swing.JMenuItem maxPrecisionXYData1;
	    private javax.swing.JMenuItem maxPrecisionXYData10;
	    private javax.swing.JMenuItem maxPrecisionXYData2;
	    private javax.swing.JMenuItem maxPrecisionXYData3;
	    private javax.swing.JMenuItem maxPrecisionXYData4;
	    private javax.swing.JMenuItem maxPrecisionXYData5;
	    private javax.swing.JMenuItem maxPrecisionXYData6;
	    private javax.swing.JMenuItem maxPrecisionXYData7;
	    private javax.swing.JMenuItem maxPrecisionXYData8;
	    private javax.swing.JMenuItem maxPrecisionXYData9;
	    private javax.swing.JTextField maxPrecisionZ;
	    private javax.swing.JMenu maxPrecisionZChList;
	    private javax.swing.JMenuItem maxPrecisionZData1;
	    private javax.swing.JMenuItem maxPrecisionZData10;
	    private javax.swing.JMenuItem maxPrecisionZData2;
	    private javax.swing.JMenuItem maxPrecisionZData3;
	    private javax.swing.JMenuItem maxPrecisionZData4;
	    private javax.swing.JMenuItem maxPrecisionZData5;
	    private javax.swing.JMenuItem maxPrecisionZData6;
	    private javax.swing.JMenuItem maxPrecisionZData7;
	    private javax.swing.JMenuItem maxPrecisionZData8;
	    private javax.swing.JMenuItem maxPrecisionZData9;
	    private javax.swing.JTextField maxRsquare;
	    private javax.swing.JMenu maxRsquareChList;
	    private javax.swing.JMenuItem maxRsquareData1;
	    private javax.swing.JMenuItem maxRsquareData10;
	    private javax.swing.JMenuItem maxRsquareData2;
	    private javax.swing.JMenuItem maxRsquareData3;
	    private javax.swing.JMenuItem maxRsquareData4;
	    private javax.swing.JMenuItem maxRsquareData5;
	    private javax.swing.JMenuItem maxRsquareData6;
	    private javax.swing.JMenuItem maxRsquareData7;
	    private javax.swing.JMenuItem maxRsquareData8;
	    private javax.swing.JMenuItem maxRsquareData9;
	    private javax.swing.JTextField maxSigmaXY;
	    private javax.swing.JMenu maxSigmaXYChList;
	    private javax.swing.JMenuItem maxSigmaXYData1;
	    private javax.swing.JMenuItem maxSigmaXYData10;
	    private javax.swing.JMenuItem maxSigmaXYData2;
	    private javax.swing.JMenuItem maxSigmaXYData3;
	    private javax.swing.JMenuItem maxSigmaXYData4;
	    private javax.swing.JMenuItem maxSigmaXYData5;
	    private javax.swing.JMenuItem maxSigmaXYData6;
	    private javax.swing.JMenuItem maxSigmaXYData7;
	    private javax.swing.JMenuItem maxSigmaXYData8;
	    private javax.swing.JMenuItem maxSigmaXYData9;
	    private javax.swing.JMenu maxSigmaZChList;
	    private javax.swing.JMenuItem maxSigmaZData1;
	    private javax.swing.JMenuItem maxSigmaZData10;
	    private javax.swing.JMenuItem maxSigmaZData2;
	    private javax.swing.JMenuItem maxSigmaZData3;
	    private javax.swing.JMenuItem maxSigmaZData4;
	    private javax.swing.JMenuItem maxSigmaZData5;
	    private javax.swing.JMenuItem maxSigmaZData6;
	    private javax.swing.JMenuItem maxSigmaZData7;
	    private javax.swing.JMenuItem maxSigmaZData8;
	    private javax.swing.JMenuItem maxSigmaZData9;
	    private javax.swing.JTextField minFrame;
	    private javax.swing.JMenu minFrameChList;
	    private javax.swing.JMenuItem minFrameData1;
	    private javax.swing.JMenuItem minFrameData10;
	    private javax.swing.JMenuItem minFrameData2;
	    private javax.swing.JMenuItem minFrameData3;
	    private javax.swing.JMenuItem minFrameData4;
	    private javax.swing.JMenuItem minFrameData5;
	    private javax.swing.JMenuItem minFrameData6;
	    private javax.swing.JMenuItem minFrameData7;
	    private javax.swing.JMenuItem minFrameData8;
	    private javax.swing.JMenuItem minFrameData9;
	    private javax.swing.JLabel minLabel;
	    private javax.swing.JLabel minLabel1;
	    private javax.swing.JLabel minLabel2;
	    private javax.swing.JLabel minLabel3;
	    private javax.swing.JLabel minLabel4;
	    private javax.swing.JTextField minPhotonCount;
	    private javax.swing.JMenu minPhotonCountChList;
	    private javax.swing.JMenuItem minPhotonCountData1;
	    private javax.swing.JMenuItem minPhotonCountData10;
	    private javax.swing.JMenuItem minPhotonCountData2;
	    private javax.swing.JMenuItem minPhotonCountData3;
	    private javax.swing.JMenuItem minPhotonCountData4;
	    private javax.swing.JMenuItem minPhotonCountData5;
	    private javax.swing.JMenuItem minPhotonCountData6;
	    private javax.swing.JMenuItem minPhotonCountData7;
	    private javax.swing.JMenuItem minPhotonCountData8;
	    private javax.swing.JMenuItem minPhotonCountData9;
	    private javax.swing.JTextField minPrecisionXY;
	    private javax.swing.JMenu minPrecisionXYChList;
	    private javax.swing.JMenuItem minPrecisionXYData1;
	    private javax.swing.JMenuItem minPrecisionXYData10;
	    private javax.swing.JMenuItem minPrecisionXYData2;
	    private javax.swing.JMenuItem minPrecisionXYData3;
	    private javax.swing.JMenuItem minPrecisionXYData4;
	    private javax.swing.JMenuItem minPrecisionXYData5;
	    private javax.swing.JMenuItem minPrecisionXYData6;
	    private javax.swing.JMenuItem minPrecisionXYData7;
	    private javax.swing.JMenuItem minPrecisionXYData8;
	    private javax.swing.JMenuItem minPrecisionXYData9;
	    private javax.swing.JTextField minPrecisionZ;
	    private javax.swing.JMenu minPrecisionZChList;
	    private javax.swing.JMenuItem minPrecisionZData1;
	    private javax.swing.JMenuItem minPrecisionZData10;
	    private javax.swing.JMenuItem minPrecisionZData2;
	    private javax.swing.JMenuItem minPrecisionZData3;
	    private javax.swing.JMenuItem minPrecisionZData4;
	    private javax.swing.JMenuItem minPrecisionZData5;
	    private javax.swing.JMenuItem minPrecisionZData6;
	    private javax.swing.JMenuItem minPrecisionZData7;
	    private javax.swing.JMenuItem minPrecisionZData8;
	    private javax.swing.JMenuItem minPrecisionZData9;
	    private javax.swing.JTextField minPtsCluster;
	    private javax.swing.JMenu minPtsClusterChList;
	    private javax.swing.JMenuItem minPtsClusterData1;
	    private javax.swing.JMenuItem minPtsClusterData10;
	    private javax.swing.JMenuItem minPtsClusterData2;
	    private javax.swing.JMenuItem minPtsClusterData3;
	    private javax.swing.JMenuItem minPtsClusterData4;
	    private javax.swing.JMenuItem minPtsClusterData5;
	    private javax.swing.JMenuItem minPtsClusterData6;
	    private javax.swing.JMenuItem minPtsClusterData7;
	    private javax.swing.JMenuItem minPtsClusterData8;
	    private javax.swing.JMenuItem minPtsClusterData9;
	    private javax.swing.JLabel minPtsLabel;
	    private javax.swing.JTextField minRsquare;
	    private javax.swing.JMenu minRsquareChList;
	    private javax.swing.JMenuItem minRsquareData1;
	    private javax.swing.JMenuItem minRsquareData10;
	    private javax.swing.JMenuItem minRsquareData2;
	    private javax.swing.JMenuItem minRsquareData3;
	    private javax.swing.JMenuItem minRsquareData4;
	    private javax.swing.JMenuItem minRsquareData5;
	    private javax.swing.JMenuItem minRsquareData6;
	    private javax.swing.JMenuItem minRsquareData7;
	    private javax.swing.JMenuItem minRsquareData8;
	    private javax.swing.JMenuItem minRsquareData9;
	    private javax.swing.JTextField minSigmaXY;
	    private javax.swing.JMenu minSigmaXYChList;
	    private javax.swing.JMenuItem minSigmaXYData1;
	    private javax.swing.JMenuItem minSigmaXYData10;
	    private javax.swing.JMenuItem minSigmaXYData2;
	    private javax.swing.JMenuItem minSigmaXYData3;
	    private javax.swing.JMenuItem minSigmaXYData4;
	    private javax.swing.JMenuItem minSigmaXYData5;
	    private javax.swing.JMenuItem minSigmaXYData6;
	    private javax.swing.JMenuItem minSigmaXYData7;
	    private javax.swing.JMenuItem minSigmaXYData8;
	    private javax.swing.JMenuItem minSigmaXYData9;
	    private javax.swing.JMenu minSigmaZChList;
	    private javax.swing.JMenuItem minSigmaZData1;
	    private javax.swing.JMenuItem minSigmaZData10;
	    private javax.swing.JMenuItem minSigmaZData2;
	    private javax.swing.JMenuItem minSigmaZData3;
	    private javax.swing.JMenuItem minSigmaZData4;
	    private javax.swing.JMenuItem minSigmaZData5;
	    private javax.swing.JMenuItem minSigmaZData6;
	    private javax.swing.JMenuItem minSigmaZData7;
	    private javax.swing.JMenuItem minSigmaZData8;
	    private javax.swing.JMenuItem minSigmaZData9;
	    private javax.swing.JTextField minimalSignal;
	    private javax.swing.JMenu minimalSignalChList;
	    private javax.swing.JMenuItem minimalSignalData1;
	    private javax.swing.JMenuItem minimalSignalData10;
	    private javax.swing.JMenuItem minimalSignalData2;
	    private javax.swing.JMenuItem minimalSignalData3;
	    private javax.swing.JMenuItem minimalSignalData4;
	    private javax.swing.JMenuItem minimalSignalData5;
	    private javax.swing.JMenuItem minimalSignalData6;
	    private javax.swing.JMenuItem minimalSignalData7;
	    private javax.swing.JMenuItem minimalSignalData8;
	    private javax.swing.JMenuItem minimalSignalData9;
	    private javax.swing.JLabel minimalSignalLabel;
	    private javax.swing.JComboBox<String> modality;
	    private javax.swing.JTextField numberOfBinsDriftCorr;
	    private javax.swing.JMenu numberOfBinsDriftCorrChList;
	    private javax.swing.JMenuItem numberOfBinsDriftCorrData1;
	    private javax.swing.JMenuItem numberOfBinsDriftCorrData10;
	    private javax.swing.JMenuItem numberOfBinsDriftCorrData2;
	    private javax.swing.JMenuItem numberOfBinsDriftCorrData3;
	    private javax.swing.JMenuItem numberOfBinsDriftCorrData4;
	    private javax.swing.JMenuItem numberOfBinsDriftCorrData5;
	    private javax.swing.JMenuItem numberOfBinsDriftCorrData6;
	    private javax.swing.JMenuItem numberOfBinsDriftCorrData7;
	    private javax.swing.JMenuItem numberOfBinsDriftCorrData8;
	    private javax.swing.JMenuItem numberOfBinsDriftCorrData9;
	    private javax.swing.JLabel numberOfBinsLabel;
	    private javax.swing.JTextField outputPixelSize;
	    private javax.swing.JMenu outputPixelSizeChList;
	    private javax.swing.JMenuItem outputPixelSizeData1;
	    private javax.swing.JMenuItem outputPixelSizeData10;
	    private javax.swing.JMenuItem outputPixelSizeData2;
	    private javax.swing.JMenuItem outputPixelSizeData3;
	    private javax.swing.JMenuItem outputPixelSizeData4;
	    private javax.swing.JMenuItem outputPixelSizeData5;
	    private javax.swing.JMenuItem outputPixelSizeData6;
	    private javax.swing.JMenuItem outputPixelSizeData7;
	    private javax.swing.JMenuItem outputPixelSizeData8;
	    private javax.swing.JMenuItem outputPixelSizeData9;
	    private javax.swing.JLabel outputPixelSizeLabel;
	    private javax.swing.JTextField outputPixelSizeZ;
	    private javax.swing.JRadioButton parallelComputation;
	    private javax.swing.JLabel particlesPerBinLabel;
	    private javax.swing.JLabel particlesPerBinLabel1;
	    private javax.swing.JLabel particlesPerBinLabel2;
	    private javax.swing.JLabel particlesPerBinLabelchAlign;
	    private javax.swing.JMenu pixelSizeChList;
	    private javax.swing.JMenuItem pixelSizeData1;
	    private javax.swing.JMenuItem pixelSizeData10;
	    private javax.swing.JMenuItem pixelSizeData2;
	    private javax.swing.JMenuItem pixelSizeData3;
	    private javax.swing.JMenuItem pixelSizeData4;
	    private javax.swing.JMenuItem pixelSizeData5;
	    private javax.swing.JMenuItem pixelSizeData6;
	    private javax.swing.JMenuItem pixelSizeData7;
	    private javax.swing.JMenuItem pixelSizeData8;
	    private javax.swing.JMenuItem pixelSizeData9;
	    private javax.swing.JButton renderImage;
	    private javax.swing.JButton resetBasicInput;
	    private javax.swing.JButton resetParameterRange;
	    private javax.swing.JButton storeSettings;
	    private javax.swing.JTextField totalGain;
	    private javax.swing.JMenu totalGainChList;
	    private javax.swing.JMenuItem totalGainData1;
	    private javax.swing.JMenuItem totalGainData10;
	    private javax.swing.JMenuItem totalGainData2;
	    private javax.swing.JMenuItem totalGainData3;
	    private javax.swing.JMenuItem totalGainData4;
	    private javax.swing.JMenuItem totalGainData5;
	    private javax.swing.JMenuItem totalGainData6;
	    private javax.swing.JMenuItem totalGainData7;
	    private javax.swing.JMenuItem totalGainData8;
	    private javax.swing.JMenuItem totalGainData9;
	    private javax.swing.JLabel totalGainLabel;
	    private javax.swing.JTextField windowWidth;
	    private javax.swing.JMenu windowWidthChList;
	    private javax.swing.JMenuItem windowWidthData1;
	    private javax.swing.JMenuItem windowWidthData10;
	    private javax.swing.JMenuItem windowWidthData2;
	    private javax.swing.JMenuItem windowWidthData3;
	    private javax.swing.JMenuItem windowWidthData4;
	    private javax.swing.JMenuItem windowWidthData5;
	    private javax.swing.JMenuItem windowWidthData6;
	    private javax.swing.JMenuItem windowWidthData7;
	    private javax.swing.JMenuItem windowWidthData8;
	    private javax.swing.JMenuItem windowWidthData9;
	    private javax.swing.JLabel windowWidthLabel;
	    // End of variables declaration                     
 }