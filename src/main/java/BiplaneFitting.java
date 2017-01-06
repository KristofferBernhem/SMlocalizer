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


// TODO: angle calculated correctly but somehow not passed. Check interpolate function for error.
import java.util.ArrayList;

import ij.ImagePlus;
import ij.WindowManager;
import ij.process.ImageProcessor;
import ij.process.ImageStatistics;


public class BiplaneFitting {
	public static ArrayList<Particle> fit(ArrayList<Particle> inputResults, int inputPixelSize, int[] totalGain)
	{
		ImagePlus image = WindowManager.getCurrentImage();
		int frameWidth 	= image.getWidth()*inputPixelSize;
		ArrayList<Particle> results = new ArrayList<Particle>();
		double[][] calibration 		= getCalibration();
		
		
		double offsetX 	= ij.Prefs.get("SMLocalizer.calibration.Biplane.finalOffsetX",0);
		double offsetY 	= ij.Prefs.get("SMLocalizer.calibration.Biplane.finalOffsetY",0);
		int gWindow 	= (int) ij.Prefs.get("SMLocalizer.calibration.Biplane.sigma",0);
		
		for (int i = 0; i < inputResults.size()-1; i++) // loop over all entries
		{
			double searchX = inputResults.get(i).x;
			double searchY = inputResults.get(i).y;
			if (inputResults.get(i).include == 1)
			{
				inputResults.get(i).include = 2; // checked.
				if (inputResults.get(i).x < frameWidth) // if on the left hand side.
				{
					/*
					 * check to see if there is another fitted object on the right side within 1 pixel + offset.
					 * if so, use the best fit of these two as dominant and calculate summed intensity within fit region
					 * for both, take ratio and store this. Center coordinate is transposed by offset if the dominant one is on the right.
					 * 
					 */
					searchX += offsetX + frameWidth;
					searchY += offsetY;
				}else{
					searchX -= - (offsetX + frameWidth);
					searchY -= - offsetY;
				}
				// find if any object are close to searchX,searchY.
				//for (int j = i + 1; j < result.size(); j++ )
				int j = i + 1;
				boolean search = true;
				//ArrayList<Particle> correctedResult = new ArrayList<Particle>();
				double z = 0;
				int photon = 0;
				while (search)
				{
					if (inputResults.get(j).channel == inputResults.get(i).channel &&  // if the same channel
							inputResults.get(j).frame == inputResults.get(i).frame)    // and same frame-
					{
						if (inputResults.get(j).y < (searchY + inputPixelSize) &&
								inputResults.get(j).y > (searchY - inputPixelSize))
						{
							if (inputResults.get(j).x < (searchX + inputPixelSize) &&
									inputResults.get(j).x > (searchX - inputPixelSize))
							{											
								inputResults.get(j).include = 2; // this entry has been covered.

								if(inputResults.get(i).x < frameWidth)
								{																							

									double photonLeft 	= getPhotons(image, inputResults.get(i).x/inputPixelSize,inputResults.get(i).y/inputPixelSize, inputResults.get(i).frame, inputResults.get(i).channel, gWindow,totalGain);
									double photonRight 	= getPhotons(image, inputResults.get(j).x/inputPixelSize,inputResults.get(j).y/inputPixelSize, inputResults.get(i).frame, inputResults.get(i).channel, gWindow,totalGain);
									photon = (int) (photonLeft + photonRight);
									z = getZ(calibration, inputResults.get(i).channel, photonLeft/photonRight);
								}else
								{
									double photonLeft 	= getPhotons(image, inputResults.get(j).x/inputPixelSize,inputResults.get(j).y/inputPixelSize, inputResults.get(i).frame, inputResults.get(i).channel, gWindow,totalGain);
									double photonRight 	= getPhotons(image, inputResults.get(i).x/inputPixelSize,inputResults.get(i).y/inputPixelSize, inputResults.get(i).frame, inputResults.get(i).channel, gWindow,totalGain);
									photon = (int) (photonLeft + photonRight);
									z = getZ(calibration, inputResults.get(i).channel, photonLeft/photonRight);							
								}
								search = false;
							}
						}
					}

					j++;

					if (j == inputResults.size() || 							// if we're at the end of the list and still have not found any other located events that match.
							inputResults.get(j).frame > inputResults.get(i).frame)	// if we're looking in the wrong frame, stop.
					{
						search = false;
						if(inputResults.get(i).x < frameWidth)
						{																							
							double photonLeft 	= getPhotons(image, inputResults.get(i).x/inputPixelSize,inputResults.get(i).y/inputPixelSize, inputResults.get(i).frame, inputResults.get(i).channel, gWindow,totalGain);
							double photonRight 	= getPhotons(image, searchX/inputPixelSize,searchY/inputPixelSize, inputResults.get(i).frame, inputResults.get(i).channel, gWindow,totalGain);
							photon = (int) (photonLeft + photonRight);
							z = getZ(calibration, inputResults.get(i).channel, photonLeft/photonRight);																
						}else
						{
							double photonLeft 	= getPhotons(image, searchX/inputPixelSize,searchY/inputPixelSize, inputResults.get(i).frame, inputResults.get(i).channel, gWindow,totalGain);
							double photonRight 	= getPhotons(image, inputResults.get(i).x/inputPixelSize,inputResults.get(i).y/inputPixelSize, inputResults.get(i).frame, inputResults.get(i).channel, gWindow,totalGain);
							photon = (int) (photonLeft + photonRight);
							z = getZ(calibration, inputResults.get(i).channel, photonLeft/photonRight);							
						}
					}
				}
				if (inputResults.get(i).r_square > inputResults.get(j).r_square)
				{
					Particle temp 	= new Particle();
					temp.channel 	= inputResults.get(i).channel;
					temp.z 		 	= z;
					temp.frame 	 	= inputResults.get(i).frame;
					temp.photons 	= photon;
					temp.x			= inputResults.get(i).x;
					temp.y			= inputResults.get(i).y;
					temp.sigma_x 	= inputResults.get(i).sigma_x; 		// fitted sigma in x direction.
					temp.sigma_y 	= inputResults.get(i).sigma_y; 		// fitted sigma in x direction.					
					temp.precision_x= inputResults.get(i).precision_x; 	// precision of fit for x coordinate.
					temp.precision_y= inputResults.get(i).precision_y; 	// precision of fit for y coordinate.
					temp.precision_z= 600 / Math.sqrt(temp.photons); 			// precision of fit for z coordinate.
					temp.r_square 	= inputResults.get(i).r_square; 	// Goodness of fit.
					temp.include	= 1; 		// If this particle should be included in analysis and plotted.
					if (temp.z != -1)
						results.add(temp);
				}else
				{
					Particle temp 	= new Particle();
					temp.channel 	= inputResults.get(j).channel;
					temp.z 		 	= z;
					temp.frame 	 	= inputResults.get(j).frame;
					temp.photons 	= photon;
					temp.x			= inputResults.get(j).x;
					temp.y			= inputResults.get(j).y;
					temp.sigma_x 	= inputResults.get(j).sigma_x; 		// fitted sigma in x direction.
					temp.sigma_y 	= inputResults.get(j).sigma_y; 		// fitted sigma in x direction.					
					temp.precision_x= inputResults.get(j).precision_x; 	// precision of fit for x coordinate.
					temp.precision_y= inputResults.get(j).precision_y; 	// precision of fit for y coordinate.
					temp.precision_z= 600 / Math.sqrt(temp.photons); 			// precision of fit for z coordinate.
					temp.r_square 	= inputResults.get(j).r_square; 	// Goodness of fit.
					temp.include	= 1; 		// If this particle should be included in analysis and plotted.
					if (temp.z != -1)
						results.add(temp);
				}

			}
		}		
		
		
		return results;
	}

	public static void calibrate(int inputPixelSize,int zStep)
	{
		ImagePlus image 		= WindowManager.getCurrentImage();
		int nFrames 			= image.getNFrames();
		if (nFrames == 1)
			nFrames 			= image.getNSlices(); 
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

		/*
		 * remove frame median from calibration file.
		 */

		for (int channel = 1; channel <= nChannels; channel++)
		{
			if (image.getNFrames() == 1)
			{
				image.setPosition(							
						channel,// channel.
						1,	// slice.
						1);		// frame.
			}
			else
			{														
				image.setPosition(
						channel,			// channel.
						1,			// slice.
						1);		// frame.
			}
			ImageProcessor IP = image.getProcessor();
			double median = 0;
			for (int i = 0; i < IP.getPixelCount(); i++)
				median += IP.get(i);
			median /= IP.getPixelCount();
			for (int frame = 1; frame <= nFrames; frame ++)
			{
				if (image.getNFrames() == 1)
				{
					image.setPosition(							
							channel,// channel.
							frame,	// slice.
							1);		// frame.
				}
				else
				{														
					image.setPosition(
							channel,			// channel.
							1,			// slice.
							frame);		// frame.
				}

				IP = image.getProcessor();
				int value = 0;
				for (int i = 0; i < IP.getPixelCount(); i++)
				{								
					value = (int)(IP.get(i)-median);
					if (value < 0)
						value = 0;
					IP.set(i, value);
				}
				//image.setProcessor(IP);
				image.updateAndDraw();
			}

		}


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

		double finalLevel 	= 0;
		double finalSigma 	= 0;
		double finalOffsetX = 0;
		double finalOffsetY = 0;		
		int gWindow 		= 5;
		int frameWidth = inputPixelSize*image.getWidth()/2;
		if (inputPixelSize < 100)
		{
			gWindow = (int) Math.ceil(500 / inputPixelSize); // 500 nm wide window.

			if (gWindow%2 == 0)
				gWindow++;	
		}
		int finalGWindow 	= gWindow;
		int loopC 			= 0;
		while (loopC < 2)
		{
			gWindow = gWindow + loopC*2; // increase window size each loop.
			for (double level = 0.7; level > 0.4; level -= 0.1)
			{
				for (double maxSigma = 2.5; maxSigma < 4; maxSigma += 0.5)
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
					result 				= TableIO.Load();
					double[][] ratio	= new double[nFrames][nChannels];
					int[][] count		= new int[nFrames][nChannels];
					/*
					 * Find offset based on center stack fits.
					 */

					double offsetX = 0;
					double offsetY = 0;
					int offsetCount = 0;


					for (int i = 0; i < result.size(); i++)
					{
						if (result.get(i).frame > ((int)nFrames/2-5) &&	// if we're on the within the 5 slizes closest to the center
								result.get(i).frame < ((int)nFrames/2+5) &&
								result.get(i).x < frameWidth) 			// and on the left side of the image.
						{
							/*
							 * find closest fit on right hand side of the image from that frame, frameWidth to the right.
							 */
							double tempOffset = frameWidth;
							double tempOffsetX = frameWidth;
							double tempOffsetY = frameWidth;
							for (int j = 0; j < result.size(); j++)
							{
								if (result.get(j).frame == result.get(i).frame &&
										result.get(j).x > frameWidth &&
										result.get(j).channel == result.get(i).channel)
								{
									double newX = result.get(i).x + frameWidth; // shift x value by the half frame width.
									double temp = Math.sqrt((newX - result.get(j).x)*(newX - result.get(j).x) + 
											(result.get(i).y - result.get(j).y)*(result.get(i).y - result.get(j).y));

									if (temp < tempOffset)
									{
										tempOffset = temp;
										tempOffsetX = newX - result.get(j).x;
										tempOffsetY = result.get(i).y - result.get(j).y;
									}
								}
							}
							offsetX += tempOffsetX; //update
							offsetY += tempOffsetY; //update
							offsetCount++;
						}											
					}
					offsetX /= offsetCount; // normalize.
					offsetY /= offsetCount; // normalize.

					/*
					 * Calculate ratio using this offset.
					 */

					for (int i = 0; i < result.size()-1; i++) // loop over all entries
					{
						double searchX = result.get(i).x;
						double searchY = result.get(i).y;
						if (result.get(i).include == 1)
						{
							result.get(i).include = 2; // checked.
							if (result.get(i).x < frameWidth) // if on the left hand side.
							{
								/*
								 * check to see if there is another fitted object on the right side within 1 pixel + offset.
								 * if so, use the best fit of these two as dominant and calculate summed intensity within fit region
								 * for both, take ratio and store this. Center coordinate is transposed by offset if the dominant one is on the right.
								 * 
								 */
								searchX += offsetX + frameWidth;
								searchY += offsetY;
							}else{
								searchX -= - (offsetX + frameWidth);
								searchY -= - offsetY;
							}
							// find if any object are close to searchX,searchY.
							//for (int j = i + 1; j < result.size(); j++ )
							int j = i + 1;
							boolean search = true;
							//ArrayList<Particle> correctedResult = new ArrayList<Particle>();
							while (search)
							{
								if (result.get(j).channel == result.get(i).channel &&  // if the same channel
										result.get(j).frame == result.get(i).frame)    // and same frame-
								{
									if (result.get(j).y < (searchY + inputPixelSize) &&
											result.get(j).y > (searchY - inputPixelSize))
									{
										if (result.get(j).x < (searchX + inputPixelSize) &&
												result.get(j).x > (searchX - inputPixelSize))
										{											
											result.get(j).include = 2; // this entry has been covered.
											if (result.get(j).r_square > result.get(i).r_square)
											{
												// for fitting, use j entry for calculations.
											}
											if(result.get(i).x < frameWidth)
											{																							

												//ratio[result.get(i).frame][result.get(i).channel-1] += (double)(result.get(i).photons) / (double)(result.get(j).photons);

												double photonLeft 	= getPhotons(image, result.get(i).x/inputPixelSize,result.get(i).y/inputPixelSize, result.get(i).frame, result.get(i).channel, gWindow,totalGain);
												double photonRight 	= getPhotons(image, result.get(j).x/inputPixelSize,result.get(j).y/inputPixelSize, result.get(i).frame, result.get(i).channel, gWindow,totalGain);												
												ratio[result.get(i).frame][result.get(i).channel-1] += photonLeft/photonRight;
												count[result.get(i).frame][result.get(i).channel-1]++;												
											}else
											{
												double photonLeft 	= getPhotons(image, result.get(j).x/inputPixelSize,result.get(j).y/inputPixelSize, result.get(i).frame, result.get(i).channel, gWindow,totalGain);
												double photonRight 	= getPhotons(image, result.get(i).x/inputPixelSize,result.get(i).y/inputPixelSize, result.get(i).frame, result.get(i).channel, gWindow,totalGain);												
												ratio[result.get(i).frame][result.get(i).channel-1] += photonLeft/photonRight;
												count[result.get(i).frame][result.get(i).channel-1]++;							
											}
											search = false;
										}
									}
								}


								j++;

								if (j == result.size() || 							// if we're at the end of the list and still have not found any other located events that match.
										result.get(j).frame > result.get(i).frame)	// if we're looking in the wrong frame, stop.
								{
									search = false;
									if(result.get(i).x < frameWidth)
									{																							

										//ratio[result.get(i).frame][result.get(i).channel-1] += (double)(result.get(i).photons) / (double)(result.get(j).photons);

										double photonLeft 	= getPhotons(image, result.get(i).x/inputPixelSize,result.get(i).y/inputPixelSize, result.get(i).frame, result.get(i).channel, gWindow,totalGain);
										double photonRight 	= getPhotons(image, searchX/inputPixelSize,searchY/inputPixelSize, result.get(i).frame, result.get(i).channel, gWindow,totalGain);												
										ratio[result.get(i).frame][result.get(i).channel-1] += photonLeft/photonRight;
										count[result.get(i).frame][result.get(i).channel-1]++;												
									}else
									{
										double photonLeft 	= getPhotons(image, searchX/inputPixelSize,searchY/inputPixelSize, result.get(i).frame, result.get(i).channel, gWindow,totalGain);
										double photonRight 	= getPhotons(image, result.get(i).x/inputPixelSize,result.get(i).y/inputPixelSize, result.get(i).frame, result.get(i).channel, gWindow,totalGain);												
										ratio[result.get(i).frame][result.get(i).channel-1] += photonLeft/photonRight;
										count[result.get(i).frame][result.get(i).channel-1]++;							
									}
								}
							}

						}
					}		
					for (int Ch = 1; Ch <= nChannels; Ch++)
					{
						for(int i = 0; i < count.length; i++)
						{
							if (count[i][Ch-1]>0)
							{
								ratio[i][Ch-1] 	/= count[i][Ch-1]; // mean ratio for this z-depth.							
							}
						}
					}


					int minLength = 40;			
					double[][] calibration = interpolateBiplane(ratio, minLength, nChannels,false);
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
							rsquare += result.get(i).r_square;
						}
						rsquare /= result.size();
						if (rsquare > meanRsquare)
						{								
							meanRsquare = rsquare;							
							finalLevel = level;
							finalSigma = maxSigma;
							finalGWindow = gWindow;
							finalOffsetX = offsetX;
							finalOffsetY = offsetY;
						}

					}			
				} // iterate over maxSigma
			} // iterate over level.
			loopC++;

		}



		ImageStatistics IMstat 	= image.getStatistics(); 
		int[] MinLevel 			= {(int) (IMstat.max*finalLevel),(int) (IMstat.max*finalLevel),(int) (IMstat.max*finalLevel),(int) (IMstat.max*finalLevel),(int) (IMstat.max*finalLevel),(int) (IMstat.max*finalLevel),(int) (IMstat.max*finalLevel),(int) (IMstat.max*finalLevel),(int) (IMstat.max*finalLevel),(int) (IMstat.max*finalLevel)};		
		Fit3D.fit(MinLevel,finalGWindow,inputPixelSize,totalGain,finalSigma);
		//localizeAndFit.run(MinLevel, finalGWindow, inputPixelSize, totalGain, selectedModel, finalSigma,"PRILM");
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
		//int id = 2;		
		double[][] ratio	= new double[nFrames][nChannels];	
		int[][] count 	  	= new int[nFrames][nChannels];

		for (int i = 0; i < result.size()-1; i++) // loop over all entries
		{
			double searchX = result.get(i).x;
			double searchY = result.get(i).y;
			if (result.get(i).include == 1)
			{
				result.get(i).include = 2; // checked.
				if (result.get(i).x < frameWidth) // if on the left hand side.
				{
					/*
					 * check to see if there is another fitted object on the right side within 1 pixel + offset.
					 * if so, use the best fit of these two as dominant and calculate summed intensity within fit region
					 * for both, take ratio and store this. Center coordinate is transposed by offset if the dominant one is on the right.
					 * 
					 */
					searchX += finalOffsetX + frameWidth;
					searchY += finalOffsetY;
				}else{
					searchX -= - (finalOffsetX + frameWidth);
					searchY -= - finalOffsetY;
				}
				// find if any object are close to searchX,searchY.
				//for (int j = i + 1; j < result.size(); j++ )
				int j = i + 1;
				boolean search = true;
				//ArrayList<Particle> correctedResult = new ArrayList<Particle>();
				while (search)
				{
					if (result.get(j).channel == result.get(i).channel &&  // if the same channel
							result.get(j).frame == result.get(i).frame)    // and same frame-
					{
						if (result.get(j).y < (searchY + 100) &&
								result.get(j).y > (searchY - 100))
						{
							if (result.get(j).x < (searchX + 100) &&
									result.get(j).x > (searchX - 100))
							{											
								result.get(j).include = 2; // this entry has been covered.
								if (result.get(j).r_square > result.get(i).r_square)
								{
									// for fitting, use j entry for calculations.
								}
								if(result.get(i).x < frameWidth)
								{																							

									//ratio[result.get(i).frame][result.get(i).channel-1] += (double)(result.get(i).photons) / (double)(result.get(j).photons);

									double photonLeft 	= getPhotons(image, result.get(i).x/inputPixelSize,result.get(i).y/inputPixelSize, result.get(i).frame, result.get(i).channel, gWindow,totalGain);
									double photonRight 	= getPhotons(image, result.get(j).x/inputPixelSize,result.get(j).y/inputPixelSize, result.get(i).frame, result.get(i).channel, gWindow,totalGain);												
									ratio[result.get(i).frame][result.get(i).channel-1] += photonLeft/photonRight;
									count[result.get(i).frame][result.get(i).channel-1]++;					

								}else
								{
									double photonLeft 	= getPhotons(image, result.get(j).x/inputPixelSize,result.get(j).y/inputPixelSize, result.get(i).frame, result.get(i).channel, gWindow,totalGain);
									double photonRight 	= getPhotons(image, result.get(i).x/inputPixelSize,result.get(i).y/inputPixelSize, result.get(i).frame, result.get(i).channel, gWindow,totalGain);												
									ratio[result.get(i).frame][result.get(i).channel-1] += photonLeft/photonRight;
									count[result.get(i).frame][result.get(i).channel-1]++;		

								}
								search = false;
							}
						}
					}


					j++;

					if (j == result.size() || 							// if we're at the end of the list and still have not found any other located events that match.
							result.get(j).frame > result.get(i).frame)	// if we're looking in the wrong frame, stop.
					{
						search = false;
						if(result.get(i).x < frameWidth)
						{																							

							//ratio[result.get(i).frame][result.get(i).channel-1] += (double)(result.get(i).photons) / (double)(result.get(j).photons);

							double photonLeft 	= getPhotons(image, result.get(i).x/inputPixelSize,result.get(i).y/inputPixelSize, result.get(i).frame, result.get(i).channel, gWindow,totalGain);
							double photonRight 	= getPhotons(image, searchX/inputPixelSize,searchY/inputPixelSize, result.get(i).frame, result.get(i).channel, gWindow,totalGain);												
							ratio[result.get(i).frame][result.get(i).channel-1] += photonLeft/photonRight;
							count[result.get(i).frame][result.get(i).channel-1]++;						

						}else
						{
							double photonLeft 	= getPhotons(image, searchX/inputPixelSize,searchY/inputPixelSize, result.get(i).frame, result.get(i).channel, gWindow,totalGain);
							double photonRight 	= getPhotons(image, result.get(i).x/inputPixelSize,result.get(i).y/inputPixelSize, result.get(i).frame, result.get(i).channel, gWindow,totalGain);												
							ratio[result.get(i).frame][result.get(i).channel-1] += photonLeft/photonRight;
							count[result.get(i).frame][result.get(i).channel-1]++;					

						}
					}
				}

			}
		}



		for (int Ch = 1; Ch <= nChannels; Ch++)
		{
			for(int i = 0; i < count.length; i++)
			{
				if (count[i][Ch-1]>0)
				{
					ratio[i][Ch-1] 	/= count[i][Ch-1]; // mean ratio for this z-depth.
					//		System.out.println(angle[i][Ch-1]);
				}
			}
		}
		int minLength = 40;			
		double[][] calibration = interpolateBiplane(ratio, minLength, nChannels,false);


		/*
		 * STORE calibration file:
		 */
		System.out.print(calibration.length);
		ij.Prefs.set("SMLocalizer.calibration.Biplane.window",finalGWindow);
		ij.Prefs.set("SMLocalizer.calibration.Biplane.sigma",finalSigma);		
		ij.Prefs.set("SMLocalizer.calibration.Biplane.height",calibration.length);
		ij.Prefs.set("SMLocalizer.calibration.Biplane.channels",nChannels);
		ij.Prefs.set("SMLocalizer.calibration.Biplane.step",zStep);
		ij.Prefs.set("SMLocalizer.calibration.Biplane.finalOffsetX",finalOffsetX);
		ij.Prefs.set("SMLocalizer.calibration.Biplane.finalOffsetY",finalOffsetY);

		for (int Ch = 1; Ch <= nChannels; Ch++)
		{
			for (int i = 0; i < calibration.length; i++)
			{
				ij.Prefs.set("SMLocalizer.calibration.Biplane.Ch"+Ch+"."+i,calibration[i][Ch-1]);
			}
		} 
		ij.Prefs.savePreferences(); // store settings. 

		double[] printout = new double[calibration.length];
		for (int i = 0; i < printout.length; i++){
			printout[i] = calibration[i][0];

		}
		correctDrift.plot(printout);


	} // calibrate.

	public static double getPhotons(ImagePlus image, double xi, double yi, int frame, int channel, int gWindow, int[] gain)
	{
		double photons = 0;

		if (image.getNFrames() == 1)
		{
			image.setPosition(							
					channel,	// channel.
					frame,		// slice.
					1);			// frame.
		}
		else
		{														
			image.setPosition(
					channel,	// channel.
					1,			// slice.
					frame);		// frame.
		}
		ImageProcessor IP = image.getProcessor();
		int x = (int) (xi - gWindow/2);
		int y = (int) (yi - gWindow/2);
		while (x <= (xi + gWindow/2) && 
				y <= (yi + gWindow/2))
		{
			photons += IP.get(x,y)/gain[channel-1];
			x++;
			if (x > (xi + gWindow/2))
			{
				x = (int)(xi - gWindow/2); // reset.
				y++;
			}
		}

		return photons;
	}

	public static double getZ (double[][] calibration, int channel, double ratio)
	{
		double z = 0;
		int idx = 0;
		while (calibration[idx][channel-1] > ratio && idx < calibration.length)
		{
			idx++;
		}
		if (idx == calibration.length -1 && ratio > calibration[idx][channel-1])
			z = -1;
		else if (calibration[idx][channel-1] == ratio)
			z = idx;
		else if (calibration[0][channel-1] == ratio)
			z = 0;
		else // interpolate
		{
			double diff = calibration[idx-1][channel-1] - calibration[idx][channel-1];
			double fraction = (ratio - calibration[idx][channel-1]) / diff;
			z = idx - 1 + fraction;
		} 					

		z *= ij.Prefs.get("SMLocalizer.calibration.Biplane.step",0); // scale.
		return z;
	}

	// returns calibration[zStep][channel]
	public static double[][] getCalibration()
	{
		int nChannels = (int)ij.Prefs.get("SMLocalizer.calibration.Biplane.channels",0);
		double[][] calibration = new double[(int)ij.Prefs.get("SMLocalizer.calibration.Biplane.height",0)][nChannels];
		for (int Ch = 1; Ch <= nChannels; Ch++)
		{
			for (int i = 0; i < calibration.length; i++)
			{
				calibration[i][Ch-1] = ij.Prefs.get("SMLocalizer.calibration.Biplane.Ch"+Ch+"."+i,0);
			}
		} 

		return calibration;
	}

	/*
	 * Used in generation of calibration file. Smoothes out fitted results, 5 point moving window mean.
	 */
	public static double[][] interpolateBiplane(double[][] result, int minLength, int nChannels,boolean printout)
	{
		int[] start = new int[nChannels];
		int[] end 	= new int[nChannels];
		double[] startValue = new double[nChannels];
		double[] endValue = new double[nChannels];		
		int maxCounter = 0;
		int channelIdx = 1;

		while (channelIdx <= nChannels)		
		{
			double[] tempVector = new double[result.length];
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
						tempVector[idx] = (result[idx][channelIdx-1]
								+ result[idx + 1][channelIdx-1] 
										+ result[idx + 2][channelIdx-1])/included;
					else
						tempVector[idx] = 0;
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
						tempVector[idx] = (result[idx - 1][channelIdx-1]
								+ result[idx][channelIdx-1]
										+ result[idx + 1][channelIdx-1] 
												+ result[idx + 2][channelIdx-1])/included;
					else
						tempVector[idx] = 0;
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
						tempVector[idx] = (result[idx - 2][channelIdx-1]
								+ result[idx - 1][channelIdx-1]
										+ result[idx][channelIdx-1] 
												+ result[idx + 1][channelIdx-1])/included;
					else
						tempVector[idx] = 0;
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
						tempVector[idx] = (result[idx - 2][channelIdx-1]
								+ result[idx - 1][channelIdx-1]
										+ result[idx][channelIdx-1])/included;
					else
						tempVector[idx] = 0;
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
						tempVector[idx] = (result[idx - 2][channelIdx-1]
								+ result[idx - 1][channelIdx-1]
										+ result[idx][channelIdx-1] 
												+ result[idx + 1][channelIdx-1]
														+ result[idx + 2][channelIdx-1])/included;
					else
						tempVector[idx] = 0;
				}

				idx++;

			} // tempVector populated.

			// find first and last positive value.
			idx = 0;
			while (idx < result.length && iterate)
			{
				if (tempVector[idx] > 0)
				{					
					startValue[channelIdx-1] = tempVector[idx];//(tempVector[idx] + tempVector[idx + 1] + tempVector[idx + 2]) / 3;
					startValue[channelIdx-1] /= 1.1;
					idx += 2;
					if(printout)
						System.out.println("startvalue: " + startValue[channelIdx-1] );
					while (iterate)
					{
						idx++;
						if (tempVector[idx] < startValue[channelIdx-1])
						{
							start[channelIdx-1] = idx;
							if(printout)
								System.out.println("start: " + start[channelIdx-1] );
							iterate = false;
						}
						if (printout)
						{
							for (int i = 0; i < 30; i++)
								System.out.println(tempVector[i] + " vs " + result[i][channelIdx-1]);
						}

					}

				}
				idx++;
			}
			idx = result.length-1;
			if(printout)
				System.out.println("idx: " + idx);
			iterate = true;
			while (idx > start[channelIdx-1] + 2 && iterate)
			{
				if (tempVector[idx] > 0)
				{
					endValue[channelIdx-1] = (tempVector[idx] + tempVector[idx - 1] + tempVector[idx - 2]) / 3;
					endValue[channelIdx-1] *= 1.1;
					idx -= 2;
					if(printout)
						System.out.println("endvalue: " + endValue[channelIdx-1] );
					while (iterate)
					{
						idx--;
						if (tempVector[idx] > endValue[channelIdx-1])
						{
							end[channelIdx-1] = idx;
							if(printout)
								System.out.println("end: " + end[channelIdx-1] );
							iterate = false;
							if (end[channelIdx-1] - start[channelIdx-1] + 1> maxCounter)
								maxCounter = end[channelIdx-1] - start[channelIdx-1] + 1;
						}

					}
				}
				idx--;
			}

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


}