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
 * TODO: Error handling if calibration file contains fewer channels than experiment.
 */



import java.util.ArrayList;
import ij.ImagePlus;
import ij.WindowManager;
import ij.process.ImageStatistics;
//TODO Look over interpolate and rename.
public class AstigmatismFitting {
	public static ArrayList<Particle> fit(ArrayList<Particle> inputResults)
	{
		ArrayList<Particle> results = new ArrayList<Particle>();
		double[][] calibration 		= getCalibration();
		double[] maxDim 			= getMaxDim();
		double[][] offset = getOffset();
		for (int i = 0; i < inputResults.size(); i++)
		{
			// check maxDim before proceeding.
			if (Math.max(inputResults.get(i).sigma_x, inputResults.get(i).sigma_y) < maxDim[inputResults.get(i).channel-1]) // if within ok range for max sigma dimension for this channel.
			{
				double z = getZ(calibration, inputResults.get(i).channel, inputResults.get(i).sigma_x/inputResults.get(i).sigma_y);
				Particle temp 	= new Particle();
				temp.channel 	= inputResults.get(i).channel;
				temp.z 		 	= z;
				temp.frame 	 	= inputResults.get(i).frame;
				temp.photons 	= inputResults.get(i).photons;
				temp.x			= inputResults.get(i).x;
				temp.y			= inputResults.get(i).y;
				temp.sigma_x 	= inputResults.get(i).sigma_x; 		// fitted sigma in x direction.
				temp.sigma_y 	= inputResults.get(i).sigma_y; 		// fitted sigma in x direction.					
				temp.precision_x= inputResults.get(i).precision_x; 	// precision of fit for x coordinate.
				temp.precision_y= inputResults.get(i).precision_y; 	// precision of fit for y coordinate.
				temp.precision_z= 600 / Math.sqrt(temp.photons); 			// precision of fit for z coordinate.
				temp.r_square 	= inputResults.get(i).r_square; 	// Goodness of fit.
				temp.include	= 1; 		// If this particle should be included in analysis and plotted.
				if(temp.z != -1 && temp.channel>1)
				{
					temp.x -= offset[0][temp.channel-1];
					temp.y -= offset[1][temp.channel-1];
					temp.z -= offset[2][temp.channel-1];
					results.add(temp);
				}
				
				if (temp.z != -1 && temp.channel==1)
					results.add(temp);
			}
		}
	//	TableIO.Store(results);
		return results;
	}

	public static void calibrate(int inputPixelSize,int zStep)
	{		
		ImagePlus image 					= WindowManager.getCurrentImage();
		int nFrames 						= image.getNFrames();
		if (nFrames == 1)
			nFrames 						= image.getNSlices(); 
		if (image.getNFrames() == 1)
		{
			image.setPosition(							
					1,			// channel.
					(int) nFrames/2,			// slice.
					1);		// frame.
		}
		else
		{														
			image.setPosition(
					1,			// channel.
					1,			// slice.
					(int) nFrames/2);		// frame.
		}
		int nChannels 			= image.getNChannels();
		int[] totalGain 		= {100,100,100,100,100,100,100,100,100,100};		
		double meanRsquare 		= 0;
		int calibrationLength 	= 0;
		boolean[][] include = new boolean[6][10];
		double[][] lb 		= new double[6][10];
		double[][] ub 		= new double[6][10];
		for (int i = 0; i < 10; i++)
		{
			include[0][i] 		= false;
			include[1][i] 		= false;
			include[2][i] 		= true;
			include[3][i] 		= false;
			include[4][i] 		= false;
			include[5][i] 		= false;


			lb[0][i]			= 0;
			lb[1][i]			= 0;
			lb[2][i]			= 0.8;
			lb[3][i]			= 0;
			lb[4][i]			= 0;
			lb[5][i]			= 0;


			ub[0][i]			= 0;
			ub[1][i]			= 0;
			ub[2][i]			= 1.0;
			ub[3][i]			= 0;
			ub[4][i]			= 0;
			ub[5][i]			= 0;

		}

		double finalLevel = 0;
		double finalSigma = 0;		
		int gWindow = 7;
		if (inputPixelSize < 100)
		{
			gWindow = (int) Math.ceil(700 / inputPixelSize); // 1500 nm wide window.

			if (gWindow%2 == 0)
				gWindow++;	
		}
		int finalGWindow = gWindow;
		int loopC = 0;

		while (loopC < 2)
		{

			for (double level = 0.7; level > 0.5; level -= 0.1)
			{
				for (double maxSigma = 4; maxSigma <= 8; maxSigma += 1)
				{									
					ImageStatistics IMstat 	= image.getStatistics();
					int[] MinLevel 			= {(int) (IMstat.max*level),(int) (IMstat.max*level),(int) (IMstat.max*level),(int) (IMstat.max*level),(int) (IMstat.max*level),(int) (IMstat.max*level),(int) (IMstat.max*level),(int) (IMstat.max*level),(int) (IMstat.max*level),(int) (IMstat.max*level)};	

					Fit3D.fit(MinLevel,gWindow,inputPixelSize,totalGain,maxSigma);
					/*
					 * clean out fits based on goodness of fit:
					 */


					cleanParticleList.run(lb,ub,include);
					cleanParticleList.delete();
					ArrayList<Particle> result = TableIO.Load();					

					for (int i = 0; i < result.size(); i++)
					{
						result.get(i).z = (result.get(i).frame-1)*zStep;
					}


					TableIO.Store(result);
					result = TableIO.Load();					
					double[][] ratio = new double[nFrames][nChannels];
					double[][] maxDim = new double[nFrames][nChannels];
					int[][] count = new int[nFrames][nChannels];
					for (int Ch = 1; Ch <= nChannels; Ch++)
					{
						for (int i = 0; i < result.size(); i++)
						{
							if (result.get(i).channel == Ch)
							{
								ratio[(int)(result.get(i).z/zStep)][Ch-1] += result.get(i).sigma_x/result.get(i).sigma_y;
								maxDim[(int)(result.get(i).z/zStep)][Ch-1] += Math.max(result.get(i).sigma_x,result.get(i).sigma_y);
								count[(int)(result.get(i).z/zStep)][Ch-1]++;
							}
						}
					}
					for (int Ch = 1; Ch <= nChannels; Ch++)
					{
						for (int i = 0; i < nFrames; i++)
						{
							if (count[i][Ch-1]>0)
							{
								ratio[i][Ch-1] /= count[i][Ch-1]; // normalize.
								maxDim[i][Ch-1] /= (count[i][Ch-1]);							
							}

						}
					}


					int minLength = 40;			
			//		System.out.println("level: " + level + " sigma: " + maxSigma + " gWindow: " + gWindow);
					double[][] calibration = makeCalibrationCurve(ratio, maxDim, minLength, nChannels, false,false, false);
					if (calibrationLength < calibration.length)
					{
						calibrationLength = calibration.length;
						meanRsquare = 0;
					}
					if (calibrationLength == calibration.length)
					{
						double rsquare = 0;
						for (int i = 0; i < result.size(); i++)
						{
							rsquare +=result.get(i).r_square;
						}
						rsquare /= result.size();
						if (rsquare > meanRsquare)
						{							
			//				System.out.println("Updated!");
							meanRsquare = rsquare;							
							finalLevel = level;
							finalSigma = maxSigma;								
							finalGWindow = gWindow;
						}

					}					
				} // iterate over maxSigma
			} // iterate over level.
			loopC++;
			gWindow = gWindow + 2; // increase window size each loop.
		}
	//	System.out.println("Final level: " + finalLevel + " sigma: " + finalSigma + " gWindow: " + finalGWindow);


		ImageStatistics IMstat 	= image.getStatistics(); 
		int[] MinLevel 			= {(int) (IMstat.max*finalLevel),(int) (IMstat.max*finalLevel),(int) (IMstat.max*finalLevel),(int) (IMstat.max*finalLevel),(int) (IMstat.max*finalLevel),(int) (IMstat.max*finalLevel),(int) (IMstat.max*finalLevel),(int) (IMstat.max*finalLevel),(int) (IMstat.max*finalLevel),(int) (IMstat.max*finalLevel)};		
		Fit3D.fit(MinLevel,finalGWindow,inputPixelSize,totalGain,finalSigma);

		/*
		 * clean out fits based on goodness of fit:
		 */


		cleanParticleList.run(lb,ub,include);
		cleanParticleList.delete();
		ArrayList<Particle> result = TableIO.Load();
		for (int i = 0; i < result.size(); i++)
		{
			result.get(i).z = (result.get(i).frame-1)*zStep;
		}
		TableIO.Store(result);
		result = TableIO.Load();
		double[][] ratio = new double[nFrames][nChannels];
		double[][] maxDim = new double[nFrames][nChannels];
		int[][] count = new int[nFrames][nChannels];
		for (int Ch = 1; Ch <= nChannels; Ch++)
		{
			for (int i = 0; i < result.size(); i++)
			{
				if (result.get(i).channel == Ch)
				{
					ratio[(int)(result.get(i).z/zStep)][Ch-1] += result.get(i).sigma_x/result.get(i).sigma_y;
					maxDim[(int)(result.get(i).z/zStep)][Ch-1] += Math.max(result.get(i).sigma_x,result.get(i).sigma_y);
					count[(int)(result.get(i).z/zStep)][Ch-1]++;
				}
			}
		}
		for (int Ch = 1; Ch <= nChannels; Ch++)
		{
			for (int i = 0; i < nFrames; i++)
			{
				if (count[i][Ch-1]>0)
				{
					ratio[i][Ch-1] /= count[i][Ch-1]; // normalize.
					maxDim[i][Ch-1] /= (count[i][Ch-1]);							
				}

			}
		}

		int minLength = 40;			
		//double[][] calibration = interpolate(ratio, minLength, nChannels);
		double[][] calibration = makeCalibrationCurve(ratio, maxDim, minLength, nChannels, false,false,true);		
		/*
		 * STORE calibration file:
		 */

		ij.Prefs.set("SMLocalizer.calibration.Astigmatism.window",finalGWindow);
		ij.Prefs.set("SMLocalizer.calibration.Astigmatism.sigma",finalSigma);		

		ij.Prefs.set("SMLocalizer.calibration.Astigmatism.height",calibration.length);
		ij.Prefs.set("SMLocalizer.calibration.Astigmatism.channels",nChannels);
		ij.Prefs.set("SMLocalizer.calibration.Astigmatism.step",zStep);
		for (int Ch = 1; Ch <= nChannels; Ch++)
		{
			for (int i = 0; i < calibration.length; i++)
			{
				ij.Prefs.set("SMLocalizer.calibration.Astigmatism.Ch"+Ch+"."+i,calibration[i][Ch-1]);
			}
		} 

		ij.Prefs.savePreferences(); // store settings.
	//	System.out.println("length: " + calibration.length);
		
		ArrayList<Particle> resultCalib = fit(result);
		TableIO.Store(resultCalib);
		/*
		 * Go through central part of the fit for each channel and calculate offset in XY for each channel.
		 */
		
		if (nChannels > 1)
		{
			int[] z = {zStep*(calibration.length/2 - 5),zStep*(calibration.length/2 + 5)}; // lower and upper bounds for z.
			double[][] offset = new double[3][nChannels-1];
			int[] counter = new int[nChannels-1];
			for (int i = 0; i < resultCalib.size(); i++)
			{
				if (resultCalib.get(i).channel == 1) // first channel.
				{
					if (resultCalib.get(i).z > z[0] && resultCalib.get(i).z < z[1]) // within center part of the calibration file.						
					{
						int ch = 2;
						while (ch <= nChannels) // reference all subsequent channels against the first one.
						{
							double particleDistance = inputPixelSize*inputPixelSize;
							int nearestNeighbor = 0;
							for (int j = i+1; j < resultCalib.size(); j++)
							{
								if (resultCalib.get(j).channel == ch)
								{
									if (resultCalib.get(i).x - resultCalib.get(j).x < inputPixelSize &&
											resultCalib.get(i).y - resultCalib.get(j).y < inputPixelSize)
									{
										double tempDist = Math.sqrt((resultCalib.get(i).x - resultCalib.get(j).x)*(resultCalib.get(i).x - resultCalib.get(j).x) + 
												(resultCalib.get(i).y - resultCalib.get(j).y)*(resultCalib.get(i).y - resultCalib.get(j).y) + 
												(resultCalib.get(i).z - resultCalib.get(j).z)*(resultCalib.get(i).z - resultCalib.get(j).z) 
												);
										if(tempDist < particleDistance)
										{
											nearestNeighbor = j;
											particleDistance = tempDist;
										}
									}
								}
							}
							counter[ch-2]++;
							offset[0][ch-2] += (resultCalib.get(nearestNeighbor).x - resultCalib.get(i).x); // remove this offset from channel ch.
							offset[1][ch-2] += (resultCalib.get(nearestNeighbor).y - resultCalib.get(i).y); // remove this offset from channel ch.
							offset[2][ch-2] += (resultCalib.get(nearestNeighbor).z - resultCalib.get(i).z); // remove this offset from channel ch.
							ch++;
						}
					}
				}
			}
			for(int i = 0; i < nChannels-1; i ++)
			{
				offset[0][i] /=counter[i];
				offset[1][i] /=counter[i];
				offset[2][i] /=counter[i];
			}
			for (int i = 1; i < nChannels; i++)
			{
				ij.Prefs.set("SMLocalizer.calibration.Astigmatism.ChOffsetX"+i,offset[0][i-1]);
				ij.Prefs.set("SMLocalizer.calibration.Astigmatism.ChOffsetY"+i,offset[1][i-1]);
				ij.Prefs.set("SMLocalizer.calibration.Astigmatism.ChOffsetZ"+i,offset[2][i-1]);
			}
			ij.Prefs.savePreferences(); // store settings.
		}else
		{
			for (int i = 1; i < 10; i++)
			{
				ij.Prefs.set("SMLocalizer.calibration.Astigmatism.ChOffsetX"+i,0);
				ij.Prefs.set("SMLocalizer.calibration.Astigmatism.ChOffsetY"+i,0);
				ij.Prefs.set("SMLocalizer.calibration.Astigmatism.ChOffsetZ"+i,0);
			}
			ij.Prefs.savePreferences(); // store settings.
		}
		
		double[] printout = new double[calibration.length];
		for (int i = 0; i < printout.length; i++)
			printout[i] = calibration[i][0];
		correctDrift.plot(printout);



	} // calibrate.

	public static double[][] getOffset()
	{
		double[][] offset = new double[3][(int) ij.Prefs.get("SMLocalizer.calibration.Astigmatism.channels",0)];
		for (int i = 1; i < offset[0].length; i++)
		{
			offset[0][i-1] = ij.Prefs.get("SMLocalizer.calibration.Astigmatism.ChOffsetX"+i,0);
			offset[1][i-1] = ij.Prefs.get("SMLocalizer.calibration.Astigmatism.ChOffsetY"+i,0);
			offset[2][i-1] = ij.Prefs.get("SMLocalizer.calibration.Astigmatism.ChOffsetZ"+i,0);
		}
		
		return offset;
	}
	
	public static double getZ (double[][] calibration, int channel, double ratio)
	{
		double z = 0;
		int idx = 1;
		while (idx < calibration.length - 1 && calibration[idx][channel-1] < ratio)
		{
			idx++;
		}
		if (idx == calibration.length -1 && ratio > calibration[idx][channel-1])
			z = -1;
		else if (calibration[idx][channel-1] == ratio)
			z = idx;
		else if (calibration[0][channel-1] == ratio)
			z = 0;
		else if (calibration[0][channel-1] > ratio)
			z = -1;
		else // interpolate
		{
			double diff = calibration[idx][channel-1] - calibration[idx - 1][channel-1];
			double fraction = (ratio - calibration[idx - 1][channel-1]) / diff;
			z = idx - 1 + fraction;
		} 		
		
		if (z != -1)
			z *= ij.Prefs.get("SMLocalizer.calibration.Astigmatism.step",0); // scale.
	//	if (z < 300)
	//		System.out.println("idx: " + idx + " z: " + z + " ratio: " + ratio);
		return z;
	}

	// returns calibration[zStep][channel]
	public static double[][] getCalibration()
	{
		int nChannels = (int)ij.Prefs.get("SMLocalizer.calibration.Astigmatism.channels",0);
		double[][] calibration = new double[(int)ij.Prefs.get("SMLocalizer.calibration.Astigmatism.height",0)][nChannels];
		for (int Ch = 1; Ch <= nChannels; Ch++)
		{
			for (int i = 0; i < calibration.length; i++)
			{
				calibration[i][Ch-1] = ij.Prefs.get("SMLocalizer.calibration.Astigmatism.Ch"+Ch+"."+i,0);
			}
		} 

		return calibration;
	}
	// returns maxDimension[channel]
	public static double[] getMaxDim()
	{
		int nChannels = (int)ij.Prefs.get("SMLocalizer.calibration.PRILM.channels",0);
		double[] maxDim = new double[nChannels];
		for (int Ch = 1; Ch <= nChannels; Ch++)
		{		
			maxDim[Ch-1] = ij.Prefs.get("SMLocalizer.calibration.Astigmatism.maxDim.Ch"+Ch,0);
		} 	
		return maxDim;
	}
	public static double[][] makeCalibrationCurve(double[][] result, double[][] selectionParameter, int minLength, int nChannels,boolean printout, boolean full, boolean store)
	{
		int[] start = new int[nChannels];
		int[] end 	= new int[nChannels];
		//double[] startValue = new double[nChannels];
		//double[] endValue = new double[nChannels];		
		int maxCounter = 0;
		int channelIdx = 1;
		double[] maxSelectionParameter = new double[nChannels];
		double leftSelectionParameter = 0;
		double rightSelectionParameter = 0;
		while (channelIdx <= nChannels)		
		{
			double[] tempVector = new double[result.length];
			double[] selectionpVector = new double[result.length];
			int idx = 0;
			boolean iterate = true;

			while (idx < result.length)
			{
				if (idx == 0)
				{
					int included = 0;
					if (result[idx][channelIdx-1] != 0)
						included++;
					if (result[idx + 1][channelIdx-1] != 0)
						included++;
					if (result[idx + 2][channelIdx-1] != 0)
						included++;
					if (included > 0)
					{
						tempVector[idx] = (result[idx][channelIdx-1]
								+ result[idx + 1][channelIdx-1] 
										+ result[idx + 2][channelIdx-1])/included;
						selectionpVector[idx] = (selectionParameter[idx][channelIdx-1]
								+ selectionParameter[idx + 1][channelIdx-1] 
										+ selectionParameter[idx + 2][channelIdx-1])/included;

					}
					else
					{
						tempVector[idx] = 0;
						selectionpVector[idx] = 0;
					}
				}else if (idx == 1)
				{
					int included = 0;					
					if (result[idx - 1][channelIdx-1] != 0)
						included++;
					if (result[idx][channelIdx-1] != 0)
						included++;
					if (result[idx + 1][channelIdx-1] != 0)
						included++;
					if (result[idx + 2][channelIdx-1] != 0)
						included++;
					if (included > 0)
					{
						tempVector[idx] = (result[idx - 1][channelIdx-1]
								+ result[idx][channelIdx-1]
										+ result[idx + 1][channelIdx-1] 
												+ result[idx + 2][channelIdx-1])/included;
						selectionpVector[idx] = (selectionParameter[idx - 1][channelIdx-1]
								+ selectionParameter[idx][channelIdx-1]
										+ selectionParameter[idx + 1][channelIdx-1] 
												+ selectionParameter[idx + 2][channelIdx-1])/included;
					}
					else
					{
						tempVector[idx] = 0;
						selectionpVector[idx] = 0;
					}
				}else if (idx == result.length - 2)
				{
					int included = 0;
					if (result[idx - 2][channelIdx-1] != 0)
						included++;
					if (result[idx - 1][channelIdx-1] != 0)
						included++;
					if (result[idx][channelIdx-1] != 0)
						included++;
					if (result[idx + 1][channelIdx-1] != 0)
						included++;
					if (included > 0)
					{
						tempVector[idx] = (result[idx - 2][channelIdx-1]
								+ result[idx - 1][channelIdx-1]
										+ result[idx][channelIdx-1] 
												+ result[idx + 1][channelIdx-1])/included;
						selectionpVector[idx] = (selectionParameter[idx - 2][channelIdx-1]
								+ selectionParameter[idx - 1][channelIdx-1]
										+ selectionParameter[idx][channelIdx-1] 
												+ selectionParameter[idx + 1][channelIdx-1])/included;
					}
					else
					{
						tempVector[idx] = 0;
						selectionpVector[idx] = 0;
					}
				}else if (idx == result.length - 1)
				{
					int included = 0;
					if (result[idx - 2][channelIdx-1] != 0)
						included++;
					if (result[idx - 1][channelIdx-1] != 0)
						included++;
					if (result[idx][channelIdx-1] != 0)
						included++;
					if (included > 0)
					{
						tempVector[idx] = (result[idx - 2][channelIdx-1]
								+ result[idx - 1][channelIdx-1]
										+ result[idx][channelIdx-1])/included;
						selectionpVector[idx] = (selectionParameter[idx - 2][channelIdx-1]
								+ selectionParameter[idx - 1][channelIdx-1]
										+ selectionParameter[idx][channelIdx-1])/included;
					}
					else
					{
						tempVector[idx] = 0;
						selectionpVector[idx] = 0;
					}
				}else if (idx < result.length - 2)
				{
					int included = 0;
					if (result[idx - 2][channelIdx-1] != 0)
						included++;
					if (result[idx - 1][channelIdx-1] != 0)
						included++;
					if (result[idx][channelIdx-1] != 0)
						included++;
					if (result[idx + 1][channelIdx-1] != 0)
						included++;
					if (result[idx + 2][channelIdx-1] != 0)
						included++;
					if (included > 0)
					{
						tempVector[idx] = (result[idx - 2][channelIdx-1]
								+ result[idx - 1][channelIdx-1]
										+ result[idx][channelIdx-1] 
												+ result[idx + 1][channelIdx-1]
														+ result[idx + 2][channelIdx-1])/included;
						selectionpVector[idx] = (selectionParameter[idx - 2][channelIdx-1]
								+ selectionParameter[idx - 1][channelIdx-1]
										+ selectionParameter[idx][channelIdx-1] 
												+ selectionParameter[idx + 1][channelIdx-1]
														+ selectionParameter[idx + 2][channelIdx-1])/included;
					}
					else
					{
						tempVector[idx] = 0;
						selectionpVector[idx] = 0;
					}
				}
				idx++;

			} // tempVector populated.

			// find first and last position for both result and selectionParameter vectors.

			idx = 1;
			int counter = 0;			
			while (idx < result.length && iterate)
			{
				if (tempVector[idx] <= 0)
				{
					counter = 0; // reset
					if (counter > minLength)
					{
						end[channelIdx-1] = idx - 1;
						iterate = false;

					}

				}

				else
				{
					if (tempVector[idx] > tempVector[idx - 1]) 
						counter++;
					if (tempVector[idx] < tempVector[idx - 1])
					{
						if (counter > minLength)
						{
							end[channelIdx-1] = idx - 1;
							iterate = false;

						}
						else
							counter = 0;
					}
				}
				if (counter == minLength)
				{
					start[channelIdx-1] = idx - minLength + 1;
				}

				idx++;
				if (idx == result.length && iterate)
				{
					end[channelIdx-1] = idx-1;
				}
			}


			/*
			 * Use selectionParameter values to make z-fitting unambiguous. 
			 */

			boolean optimize = true;
			while(optimize)
			{				
				boolean verify = true;
				leftSelectionParameter = selectionParameter[start[channelIdx-1]][channelIdx-1];
				rightSelectionParameter= selectionParameter[end[channelIdx-1]][channelIdx-1];
				int tempIdx = start[channelIdx-1]-1;
				while(verify)
				{
					if(leftSelectionParameter < selectionParameter[tempIdx][channelIdx-1] || selectionParameter[tempIdx][channelIdx-1] == 0)
					{
						tempIdx--;
					}
					else if(leftSelectionParameter >= selectionParameter[tempIdx][channelIdx-1])
					{
						verify = false;
					}
					if(tempIdx < 0)
						verify = false;
				}
				if (tempIdx > -1)
				{
					start[channelIdx-1]++;
				}else
					optimize = false;
				if ( end[channelIdx-1]- start[channelIdx-1] + 1 < minLength)
					optimize = false;
			}	
			optimize = true;
			while(optimize)
			{				
				boolean verify = true;
				leftSelectionParameter = selectionParameter[start[channelIdx-1]][channelIdx-1];
				rightSelectionParameter = selectionParameter[end[channelIdx-1]][channelIdx-1];

				int tempIdx = end[channelIdx-1]+1;
				if (tempIdx >= selectionParameter.length-1)
					{
						optimize = false;
						verify = false;
					}
				while(verify)
				{
					if(rightSelectionParameter < selectionParameter[tempIdx][channelIdx-1] || selectionParameter[tempIdx][channelIdx-1] == 0)
					{
						tempIdx++;
					}
					else if(rightSelectionParameter >= selectionParameter[tempIdx][channelIdx-1])
					{
						verify = false;
					}
					if(tempIdx == selectionParameter.length-1)
						verify = false;
				}
				if (tempIdx < selectionParameter.length-1)
				{
					end[channelIdx-1]--;
				}else
					optimize = false;
				if ( end[channelIdx-1]- start[channelIdx-1] + 1 < minLength)
					optimize = false;
			}	

			maxSelectionParameter[channelIdx-1] = Math.min(leftSelectionParameter ,  rightSelectionParameter); // find the lowest common denominator. apply it at cost of a few 10s of nm z-depth.
			if (leftSelectionParameter == maxSelectionParameter[channelIdx-1]) // change end
			{
				optimize = true;
				while(optimize)
				{
					end[channelIdx-1]--;
					rightSelectionParameter= selectionParameter[end[channelIdx-1]][channelIdx-1];
					if (rightSelectionParameter <= maxSelectionParameter[channelIdx-1])
						optimize = false;
				}
			}else // change start 
			{
				optimize = true;
				while(optimize)
				{
					start[channelIdx-1]++;
					leftSelectionParameter = selectionParameter[start[channelIdx-1]][channelIdx-1];
					if (leftSelectionParameter <= maxSelectionParameter[channelIdx-1])
						optimize = false;
				}
			}
			if(printout)
			{				
				System.out.println("Selection: " + maxSelectionParameter[channelIdx-1]);
			}
			channelIdx++;
		}
		if (store)
		{

			for (int Ch = 1; Ch <= nChannels; Ch++)
			{				
				ij.Prefs.set("SMLocalizer.calibration.Astigmatism.maxDim.Ch"+Ch,maxSelectionParameter[Ch-1]);				
			} 			
			ij.Prefs.savePreferences(); // store settings.
		}

		if(printout)
		{				
			channelIdx--;
			System.out.println("start: " + start[channelIdx-1] + " end: " + end[channelIdx-1] + " length: " + (end[channelIdx-1] - start[channelIdx-1] + 1));
		}

		maxCounter = end[0] - start[0] + 1;
		if (full)
		{
			end[0] = result.length-1;
			start[0] = 0;
			maxCounter = end[0] - start[0] + 1;	
		}
		if (maxCounter >= minLength)
		{
			double[][] calibration = new double[maxCounter][nChannels];
			channelIdx = 1;

			while (channelIdx <= nChannels)
			{
				int idx = start[channelIdx-1];
				int count = 0;
				while (idx <= end[channelIdx-1])				
				{
					// 5 point smoothing:
					if (idx == start[channelIdx-1])
					{
						int included = 0;
						if (result[idx][channelIdx-1] != 0)
							included++;
						if (result[idx + 1][channelIdx-1] != 0)
							included++;
						if (result[idx + 2][channelIdx-1] != 0)
							included++;
						if (included > 0)
							calibration[count][channelIdx-1] = (result[idx][channelIdx-1]
									+ result[idx + 1][channelIdx-1] 
											+ result[idx + 2][channelIdx-1])/included;
						else
							calibration[count][channelIdx-1] = 0;
						//						calibration[count][channelIdx-1] = (result[idx][channelIdx-1]
						//								+ result[idx + 1][channelIdx-1] 
						//										+ result[idx + 2][channelIdx-1])/3;
					}else if (idx == start[channelIdx-1] + 1)
					{
						int included = 0;					
						if (result[idx - 1][channelIdx-1] != 0)
							included++;
						if (result[idx][channelIdx-1] != 0)
							included++;
						if (result[idx + 1][channelIdx-1] != 0)
							included++;
						if (result[idx + 2][channelIdx-1] != 0)
							included++;
						if (included > 0)
							calibration[count][channelIdx-1] = (result[idx - 1][channelIdx-1]
									+ result[idx][channelIdx-1]
											+ result[idx + 1][channelIdx-1] 
													+ result[idx + 2][channelIdx-1])/included;
						else
							calibration[count][channelIdx-1] = 0;
						//						calibration[count][channelIdx-1] = (result[idx - 1][channelIdx-1]
						//								+ result[idx][channelIdx-1]
						//										+ result[idx + 1][channelIdx-1] 
						//												+ result[idx + 2][channelIdx-1])/4;
					}else if (idx == end[channelIdx-1] - 1)
					{
						int included = 0;
						if (result[idx - 2][channelIdx-1] != 0)
							included++;
						if (result[idx - 1][channelIdx-1] != 0)
							included++;
						if (result[idx][channelIdx-1] != 0)
							included++;
						if (result[idx + 1][channelIdx-1] != 0)
							included++;
						if (included > 0)
							calibration[count][channelIdx-1] = (result[idx - 2][channelIdx-1]
									+ result[idx - 1][channelIdx-1]
											+ result[idx][channelIdx-1] 
													+ result[idx + 1][channelIdx-1])/included;
						else
							calibration[count][channelIdx-1] = 0;
						//						calibration[count][channelIdx-1] = (result[idx - 2][channelIdx-1]
						//								+ result[idx - 1][channelIdx-1]
						//										+ result[idx][channelIdx-1] 
						//												+ result[idx + 1][channelIdx-1])/4;
					}else if (idx == end[channelIdx-1])
					{
						int included = 0;
						if (result[idx - 2][channelIdx-1] != 0)
							included++;
						if (result[idx - 1][channelIdx-1] != 0)
							included++;
						if (result[idx][channelIdx-1] != 0)
							included++;
						if (included > 0)
							calibration[count][channelIdx-1] = (result[idx - 2][channelIdx-1]
									+ result[idx - 1][channelIdx-1]
											+ result[idx][channelIdx-1])/included;
						else
							calibration[count][channelIdx-1] = 0;
						//						calibration[count][channelIdx-1] = (result[idx - 2][channelIdx-1]
						//								+ result[idx - 1][channelIdx-1]
						//										+ result[idx][channelIdx-1])/3;
					}else if (idx < end[channelIdx-1] - 1)
					{
						int included = 0;
						if (result[idx - 2][channelIdx-1] != 0)
							included++;
						if (result[idx - 1][channelIdx-1] != 0)
							included++;
						if (result[idx][channelIdx-1] != 0)
							included++;
						if (result[idx + 1][channelIdx-1] != 0)
							included++;
						if (result[idx + 2][channelIdx-1] != 0)
							included++;
						if (included > 0)
							calibration[count][channelIdx-1] = (result[idx - 2][channelIdx-1]
									+ result[idx - 1][channelIdx-1]
											+ result[idx][channelIdx-1] 
													+ result[idx + 1][channelIdx-1]
															+ result[idx + 2][channelIdx-1])/included;
						else
							calibration[count][channelIdx-1] = 0;
						//						calibration[count][channelIdx-1] = (result[idx - 2][channelIdx-1]
						//								+ result[idx - 1][channelIdx-1]
						//										+ result[idx][channelIdx-1] 
						//												+ result[idx + 1][channelIdx-1]
						//														+ result[idx + 2][channelIdx-1])/5;
					}
					count++;
					idx++;
				}
				channelIdx++;
			}
			return calibration;
		}

		else
		{
			double[][] calibration = new double[maxCounter][nChannels];
			return calibration;
		}
	}

	public static double[][] makeCalibrationCurve(double[][] result, int minLength, int nChannels)
	{
		int[] start = new int[nChannels];
		int[] end 	= new int[nChannels];
		int counter = 0;
		int maxCounter = 0;
		int channelIdx = 1;
		while (channelIdx <= nChannels)
		{
			int idx = 0;
			boolean iterate = true;
			while (idx < result.length && iterate)
			{
				if (result[idx][channelIdx-1] > 0)
				{
					counter++;
					if (counter == minLength) // if we've passed the set number of points.
						start[channelIdx-1] = idx - minLength + 1;
					if (counter > minLength)
						end[channelIdx-1] = idx;
				}
				else if (result[idx][channelIdx-1] <= 0) 
				{
					if (counter < minLength)
						counter = 0;
					if (counter >= minLength)
						iterate = false;
				}
				idx++;
			}
			if (counter > maxCounter)
				maxCounter = counter;

			channelIdx++;
		}	
		if (maxCounter >= minLength)
		{


			double[][] calibration = new double[maxCounter][nChannels];
			channelIdx = 1;

			while (channelIdx <= nChannels)
			{
				int idx = start[channelIdx-1];
				int count = 0;
				while (idx <= end[channelIdx-1])				
				{
					// 5 point smoothing:
					if (idx == start[channelIdx-1])
					{
						calibration[count][channelIdx-1] = (result[idx][channelIdx-1]
								+ result[idx + 1][channelIdx-1] 
										+ result[idx + 2][channelIdx-1])/3;
					}else if (idx == start[channelIdx-1] + 1)
					{
						calibration[count][channelIdx-1] = (result[idx - 1][channelIdx-1]
								+ result[idx][channelIdx-1]
										+ result[idx + 1][channelIdx-1] 
												+ result[idx + 2][channelIdx-1])/4;
					}else if (idx == end[channelIdx-1] - 1)
					{
						calibration[count][channelIdx-1] = (result[idx - 2][channelIdx-1]
								+ result[idx - 1][channelIdx-1]
										+ result[idx][channelIdx-1] 
												+ result[idx + 1][channelIdx-1])/4;
					}else if (idx == end[channelIdx-1])
					{
						calibration[count][channelIdx-1] = (result[idx - 2][channelIdx-1]
								+ result[idx - 1][channelIdx-1]
										+ result[idx][channelIdx-1])/3;
					}else if (idx < end[channelIdx-1] - 1)
					{
						calibration[count][channelIdx-1] = (result[idx - 2][channelIdx-1]
								+ result[idx - 1][channelIdx-1]
										+ result[idx][channelIdx-1] 
												+ result[idx + 1][channelIdx-1]
														+ result[idx + 2][channelIdx-1])/5;
					}
					count++;
					idx++;
				}
				channelIdx++;
			}
			return calibration;
		}

		else
		{
			double[][] calibration = new double[maxCounter][nChannels];
			return calibration;
		}
	}


}