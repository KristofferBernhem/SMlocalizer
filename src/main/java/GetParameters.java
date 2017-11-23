import ij.IJ;
public class GetParameters {

	double[][] ub;
	double[][] lb;
	int pixelSize;
	int[] totalGain;
	int[] minSignal;
	int[] windowWidth;
	boolean[] doRender;
	boolean doGaussianSmoothing;
	boolean doDriftCorrect;
	boolean doChannelAlign;
	boolean[] doMinimalSignalChList;
	boolean[] doWindowWidth;
	boolean[] doClusterAnalysis;
	double[] epsilon;
	int[] minPtsCluster;
	int[] outputPixelSize;
	int[][] driftCorrShift;
	int[] numberOfBinsDriftCorr;
	int[][] chAlignShift;
	boolean[][] includeParameters;

public GetParameters()
{	
	this.ub= new double[7][10]; // upper bound.
	this.lb = new double[7][10]; // lower bound.
	this.pixelSize = 100;
	this.totalGain = new int[10];
	this.minSignal = new int[10];
	this.windowWidth = new int[10];;
	this.doRender = new boolean[10];
	this.doGaussianSmoothing = false;
	this.doDriftCorrect = true;
	this.doChannelAlign = false;
	this.doMinimalSignalChList = new boolean[10];
	this.doWindowWidth = new boolean[10];
	this.doClusterAnalysis = new boolean[10];
	this.epsilon = new double[10];
	this.minPtsCluster = new int[10];
	this.outputPixelSize = new int[10];;
	this.driftCorrShift = new int[2][10];;
	this.numberOfBinsDriftCorr = new int[10];;
	this.chAlignShift = new int[2][10];;
	this.includeParameters = new boolean[7][10];;
}
	private void initiate() // initiate default if no values exists, finalizes with calling the get function.
	{
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
		ij.Prefs.set("SMLocalizer.calibration.Astigmatism.window",15);
		ij.Prefs.set("SMLocalizer.calibration.Astigmatism.sigma",4);		
		ij.Prefs.set("SMLocalizer.calibration.Astigmatism.height",0);
		ij.Prefs.set("SMLocalizer.calibration.Astigmatism.channels",1);
		ij.Prefs.set("SMLocalizer.calibration.Astigmatism.step",0);
		ij.Prefs.set("SMLocalizer.calibration.Astigmatism.Ch1.0",0);
		ij.Prefs.set("SMLocalizer.calibration.Astigmatism.maxDim.Ch0",0);		
		ij.Prefs.savePreferences(); // store settings.
		String storeName = "default";
		ij.Prefs.set("SMLocalizer.settingsEntries", 1);
		ij.Prefs.set("SMLocalizer.settingsName"+1, storeName); // add storename


		ij.Prefs.set("SMLocalizer.CurrentSetting",storeName);



		ij.Prefs.set("SMLocalizer.settings."+storeName+
				".doDriftCorrect.",1);



		ij.Prefs.set("SMLocalizer.settings."+storeName+
				".doChannelAlign.",0);		

		// pixel size XY
		ij.Prefs.set("SMLocalizer.settings."+storeName+
				".pixelSize", 
				"5");
		// pixel size Z
		ij.Prefs.set("SMLocalizer.settings."+storeName+
				".pixelSizeZ", 
				"10");
		// gaussian smoothing
		ij.Prefs.set("SMLocalizer.settings."+storeName+
				".doGaussianSmoothing.",0);

		for (int Ch = 0; Ch < 10; Ch++){
			/*
			 *   Basic input settings
			 */		    		                           		   

			// total gain
			ij.Prefs.set("SMLocalizer.settings."+storeName+
					".totaGain."+Ch, 
					"100");
			// minimal signal
			ij.Prefs.set("SMLocalizer.settings."+storeName+
					".minimalSignal."+Ch, 
					"800");
			// minimal signal
			ij.Prefs.set("SMLocalizer.settings."+storeName+
					".doMinimalSignal."+Ch, 
					"0");
			// filter window size
			ij.Prefs.set("SMLocalizer.settings."+storeName+
					".windowWidth."+Ch, 
					"101");
			// filter window size
			ij.Prefs.set("SMLocalizer.settings."+storeName+
					".doWindowWidth."+Ch, 
					"0");


			/*
			 *       Cluster analysis settings
			 */
			// min pixel distance
			ij.Prefs.set("SMLocalizer.settings."+storeName+
					".doClusterAnalysis."+Ch, 
					"0");
			// min pixel distance
			ij.Prefs.set("SMLocalizer.settings."+storeName+
					".epsilon."+Ch, 
					"10");
			// min pixel distance
			ij.Prefs.set("SMLocalizer.settings."+storeName+
					".minPtsCluster."+Ch, 
					"5");		    


			/*
			 *       Render image settings.
			 */
			// render image		
			ij.Prefs.set("SMLocalizer.settings."+storeName+
					".doRenderImage."+Ch,
					"1");		


			/*
			 * store parameter settings:
			 */		    		   					
			// photon count
			ij.Prefs.set("SMLocalizer.settings."+storeName+
					".doPotonCount."+Ch, 
					"0");
			ij.Prefs.set("SMLocalizer.settings."+storeName+
					".minPotonCount."+Ch, 
					"100");
			ij.Prefs.set("SMLocalizer.settings."+storeName+
					".maxPotonCount."+Ch, 
					"10000");

			// Sigma XY        		    
			ij.Prefs.set("SMLocalizer.settings."+storeName+
					".doSigmaXY."+Ch, 
					"0");
			ij.Prefs.set("SMLocalizer.settings."+storeName+
					".minSigmaXY."+Ch, 
					"100");
			ij.Prefs.set("SMLocalizer.settings."+storeName+
					".maxSigmaXY."+Ch, 
					"200");

			// Rsquare        
			ij.Prefs.set("SMLocalizer.settings."+storeName+
					".doRsquare."+Ch, 
					"1");
			ij.Prefs.set("SMLocalizer.settings."+storeName+
					".minRsquare."+Ch, 
					"0.85");
			ij.Prefs.set("SMLocalizer.settings."+storeName+
					".maxRsquare."+Ch, 
					"1.0");

			// Precision XY
			ij.Prefs.set("SMLocalizer.settings."+storeName+
					".doPrecisionXY."+Ch, 
					"0");
			ij.Prefs.set("SMLocalizer.settings."+storeName+
					".minPrecisionXY."+Ch, 
					"5");
			ij.Prefs.set("SMLocalizer.settings."+storeName+
					".maxPrecisionXY."+Ch, 
					"50");

			// Precision Z
			ij.Prefs.set("SMLocalizer.settings."+storeName+
					".doPrecisionZ."+Ch, 
					"0");
			ij.Prefs.set("SMLocalizer.settings."+storeName+
					".minPrecisionZ."+Ch, 
					"5");
			ij.Prefs.set("SMLocalizer.settings."+storeName+
					".maxPrecisionZ."+Ch, 
					"75");
			// Frame
			ij.Prefs.set("SMLocalizer.settings."+storeName+
					".doFrame."+Ch, 
					"0");
			ij.Prefs.set("SMLocalizer.settings."+storeName+
					".minFrame."+Ch, 
					"1");
			ij.Prefs.set("SMLocalizer.settings."+storeName+
					".maxFrame."+Ch, 
					"100000");
			// Z
			ij.Prefs.set("SMLocalizer.settings."+storeName+
					".doZ."+Ch, 
					"0");
			ij.Prefs.set("SMLocalizer.settings."+storeName+
					".minZ."+Ch, 
					"-1000");
			ij.Prefs.set("SMLocalizer.settings."+storeName+
					".maxZ."+Ch, 
					"1000");
			/*
			 *   Drift and channel correct settings
			 */

			// drift correction shift
			ij.Prefs.set("SMLocalizer.settings."+storeName+
					".driftCorrShiftXY."+Ch, 
					"100");
			ij.Prefs.set("SMLocalizer.settings."+storeName+
					".driftCorrShiftZ."+Ch, 
					"100");

			// number of drift bins
			ij.Prefs.set("SMLocalizer.settings."+storeName+
					".driftCorrBin."+Ch, 
					"25");

			// channel align shift
			ij.Prefs.set("SMLocalizer.settings."+storeName+
					".chAlignShiftXY."+Ch, 
					"150");
			ij.Prefs.set("SMLocalizer.settings."+storeName+
					".chAlignShiftZ."+Ch, 
					"150");	

			ij.Prefs.set("SMLocalizer.settings."+storeName+
					".correlative."+Ch, 
					"1");
			ij.Prefs.set("SMLocalizer.settings."+storeName+
					".chromatic."+Ch, 
					"1");
			ij.Prefs.set("SMLocalizer.settings."+storeName+
					".fiducials."+Ch, 
					"0");	 			 
		}



		ij.Prefs.savePreferences(); // store settings. 
	}

	public void get() // load all values.
	{

		String loadThis = ij.Prefs.get("SMLocalizer.CurrentSetting", ""); // get the latest settings.
		if (loadThis.equals("")) // if there is no data to load.
		{
			initiate();
			loadThis = ij.Prefs.get("SMLocalizer.CurrentSetting", ""); // get the latest settings.
		}
		try
		{
			ij.Prefs.set("SMLocalizer.CurrentSetting", loadThis); // current.
			if (ij.Prefs.get("SMLocalizer.settings."+loadThis+
					".doDriftCorrect.",1) == 1)
				this.doDriftCorrect = true;
			else
				this.doDriftCorrect = false; 
			if (ij.Prefs.get("SMLocalizer.settings."+loadThis+
					".doChannelAlign.",1) == 1)
				this.doChannelAlign = true;
			else
				this.doChannelAlign = false;

			// pixel size XY
			this.outputPixelSize[0] = Integer.parseInt(
					ij.Prefs.get("SMLocalizer.settings."+loadThis+
							".pixelSize", 
							""));
			// pixel size Z
			this.outputPixelSize[1] = Integer.parseInt(
					ij.Prefs.get("SMLocalizer.settings."+loadThis+
							".pixelSizeZ", 
							""));
			// gaussian smoothing
			if(ij.Prefs.get("SMLocalizer.settings."+loadThis+
					".doGaussianSmoothing.",1)== 1)
				this.doGaussianSmoothing	= true;
			else
				this.doGaussianSmoothing = false;


			for (int Ch = 0; Ch < 10; Ch++){
				/*
				 *   Basic input settings
				 */		    		                           		   

				// total gain
				this.totalGain[Ch]=				Integer.parseInt(ij.Prefs.get("SMLocalizer.settings."+loadThis+".totaGain."+Ch,""));
				//this.totalGain[Ch] = Integer.getInteger(
				//		ij.Prefs.get("SMLocalizer.settings."+loadThis+".totaGain."+Ch,""));		   

				if(ij.Prefs.get("SMLocalizer.settings."+loadThis+
						".doMinimalSignal.",1)== 1)
					this.doMinimalSignalChList[Ch]	= true;
				else
					this.doMinimalSignalChList[Ch] = false;


				// minimal signal
				this.minSignal[Ch] = Integer.parseInt(
						ij.Prefs.get("SMLocalizer.settings."+loadThis+
								".minimalSignal."+Ch, 
								""));

				// filter window size
				this.windowWidth[Ch] = Integer.parseInt(
						ij.Prefs.get("SMLocalizer.settings."+loadThis+
								".windowWidth."+Ch, 
								""));
				if(ij.Prefs.get("SMLocalizer.settings."+loadThis+
						".doWindowWidth.",1)== 1)
					this.doWindowWidth[Ch]	= true;
				else
					this.doWindowWidth[Ch] = false;

				/*
				 *       Cluster analysis settings
				 */
				// min pixel distance


				if(ij.Prefs.get("SMLocalizer.settings."+loadThis+
						".doClusterAnalysis.",1)== 1)
					this.doClusterAnalysis[Ch]	= true;
				else
					this.doClusterAnalysis[Ch] = false;


				// min pixel distance		

				this.epsilon[Ch] = Integer.parseInt(
						ij.Prefs.get("SMLocalizer.settings."+loadThis+
								".epsilon."+Ch, 
								""));
				// min pixel distance				
				this.minPtsCluster[Ch] = Integer.parseInt(
						ij.Prefs.get("SMLocalizer.settings."+loadThis+
								".minPtsCluster."+Ch, 
								""));  		    
				/*
				 *       Render image settings.
				 */
				// render image
				if(ij.Prefs.get("SMLocalizer.settings."+loadThis+
						".doRenderImage."+Ch,1) == 1)
					this.doRender[Ch]	= true;
				else
					this.doRender[Ch] = false;
				/*
				 * store parameter settings:
				 */		    	

				// photon count			
				if(ij.Prefs.get("SMLocalizer.settings."+loadThis+
						".doPotonCount."+Ch,1) == 1)
					this.includeParameters[0][Ch]	= true;
				else
					this.includeParameters[0][Ch] = false;

				this.lb[0][Ch] = Double.parseDouble(
						ij.Prefs.get("SMLocalizer.settings."+loadThis+
								".minPotonCount."+Ch, 
								""));  
				this.ub[0][Ch] = Double.parseDouble(
						ij.Prefs.get("SMLocalizer.settings."+loadThis+
								".maxPotonCount."+Ch, 
								""));  		    

				// Sigma XY  
				if(ij.Prefs.get("SMLocalizer.settings."+loadThis+
						".doSigmaXY."+Ch,1) == 1)
					this.includeParameters[1][Ch]	= true;
				else
					this.includeParameters[1][Ch] = false;

				this.lb[1][Ch] = Double.parseDouble(
						ij.Prefs.get("SMLocalizer.settings."+loadThis+
								".minSigmaXY."+Ch, 
								""));  
				this.ub[1][Ch] = Double.parseDouble(
						ij.Prefs.get("SMLocalizer.settings."+loadThis+
								".maxSigmaXY."+Ch, 
								""));  		    								

				// Rsquare  
				if(ij.Prefs.get("SMLocalizer.settings."+loadThis+
						".doRsquare."+Ch,1) == 1)
					this.includeParameters[2][Ch]	= true;
				else
					this.includeParameters[2][Ch] = false;

				this.lb[2][Ch] = Double.parseDouble(
						ij.Prefs.get("SMLocalizer.settings."+loadThis+
								".minRsquare."+Ch, 
								""));  
				this.ub[2][Ch] = Double.parseDouble(
						ij.Prefs.get("SMLocalizer.settings."+loadThis+
								".maxRsquare."+Ch, 
								""));  		    

				// Precision XY
				if(ij.Prefs.get("SMLocalizer.settings."+loadThis+
						".doPrecisionXY."+Ch,1) == 1)
					this.includeParameters[3][Ch]	= true;
				else
					this.includeParameters[3][Ch] = false;

				this.lb[3][Ch] = Double.parseDouble(
						ij.Prefs.get("SMLocalizer.settings."+loadThis+
								".minPrecisionXY."+Ch, 
								""));  
				this.ub[3][Ch] = Double.parseDouble(
						ij.Prefs.get("SMLocalizer.settings."+loadThis+
								".maxPrecisionXY."+Ch, 
								""));  		    

				// Precision Z
				if(ij.Prefs.get("SMLocalizer.settings."+loadThis+
						".doPrecisionZ."+Ch,1) == 1)
					this.includeParameters[4][Ch]	= true;
				else
					this.includeParameters[4][Ch] = false;

				this.lb[4][Ch] = Double.parseDouble(
						ij.Prefs.get("SMLocalizer.settings."+loadThis+
								".minPrecisionZ."+Ch, 
								""));  
				this.ub[4][Ch] = Double.parseDouble(
						ij.Prefs.get("SMLocalizer.settings."+loadThis+
								".maxPrecisionZ."+Ch, 
								""));  		

				// Frame
				if(ij.Prefs.get("SMLocalizer.settings."+loadThis+
						".doFrame."+Ch,1)== 1)
					this.includeParameters[5][Ch]	= true;
				else
					this.includeParameters[5][Ch] = false;

				this.lb[5][Ch] = Integer.parseInt(
						ij.Prefs.get("SMLocalizer.settings."+loadThis+
								".minFrame."+Ch, 
								""));  
				this.ub[5][Ch] = Integer.parseInt(
						ij.Prefs.get("SMLocalizer.settings."+loadThis+
								".maxFrame."+Ch, 
								""));  		

				// Z
				if(ij.Prefs.get("SMLocalizer.settings."+loadThis+
						".doZ."+Ch,1)== 1)
					this.includeParameters[6][Ch]= true;
				else
					this.includeParameters[6][Ch] = false;

				this.lb[6][Ch] = Integer.parseInt(
						ij.Prefs.get("SMLocalizer.settings."+loadThis+
								".minZ."+Ch, 
								""));  
				this.ub[6][Ch] = Integer.parseInt(
						ij.Prefs.get("SMLocalizer.settings."+loadThis+
								".maxZ."+Ch, 
								""));  	

				/*
				 *   Drift and channel correct settings
				 */

				// drift correction shift
				this.driftCorrShift[0][Ch] = Integer.parseInt(
						ij.Prefs.get("SMLocalizer.settings."+loadThis+
								".driftCorrShiftXY."+Ch, 
								""));  
				this.driftCorrShift[1][Ch] = Integer.parseInt(
						ij.Prefs.get("SMLocalizer.settings."+loadThis+
								".driftCorrShiftZ."+Ch,
								""));			    
				// number of drift bins
				this.numberOfBinsDriftCorr[Ch] = Integer.parseInt(
						ij.Prefs.get("SMLocalizer.settings."+loadThis+
								".driftCorrBin."+Ch,
								""));	
				// channel align shift
				this.chAlignShift[0][Ch] = Integer.parseInt(
						ij.Prefs.get("SMLocalizer.settings."+loadThis+
								".chAlignShiftXY."+Ch,
								""));
				this.chAlignShift[1][Ch] = Integer.parseInt(
						ij.Prefs.get("SMLocalizer.settings."+loadThis+
								".chAlignShiftZ."+Ch,
								""));	 


			} // set each channel.
		}
		finally
		{
			
		}
	}

	public void set(String[] inputParameters) // set values to a specific channel.
	{

	}


	public static void main(final String... args) 
	{
		// create the ImageJ application context with all available services				
		final ij.ImageJ ij = new ij.ImageJ();
		ij.setVisible(true);
		Class<?> clazz = GetParameters.class;
		String url = clazz.getResource("/" + clazz.getName().replace('.', '/') + ".class").toString();
		String pluginsDir = url.substring("file:".length(), url.length() - clazz.getName().length() - ".class".length());
		System.setProperty("plugins.dir", pluginsDir);				
		IJ.runPlugIn(clazz.getName(), "");
	}

	/*
	@Override
	public void run(String arg0) {

		GetParameters_ parameters = new GetParameters_();
		parameters.get();
		System.out.println(parameters.pixelSize);
		String args =  ij.Macro.getOptions(); // get macro input.
		args = args.replaceAll("\\s+","");		// remove spaces, line separators
		String[] inputParameters = args.split(",");	// split based on ","
		if (inputParameters.length == 0)
		{
			get();
		}else if (inputParameters.length == 20) // verify number.
		{
			set(inputParameters);
		}
		// switch based on input parameter count, 0 or full set to modify a channel.

	} */

}
