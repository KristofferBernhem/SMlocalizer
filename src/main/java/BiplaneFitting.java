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



import java.awt.Color;
import java.util.ArrayList;

import ij.ImagePlus;
import ij.WindowManager;
import ij.gui.Plot;
import ij.plugin.filter.Analyzer;
import ij.process.ImageProcessor;
import ij.process.ImageStatistics;

public class BiplaneFitting {
	public static ArrayList<Particle> fit(ArrayList<Particle> inputResults, int inputPixelSize, int[] totalGain)
	{
		ImagePlus image 			= WindowManager.getCurrentImage();				// load experimental image data for z calculations.
		int frameWidth 				= (int) (0.5*image.getWidth()*inputPixelSize);	// output frame width.
		ArrayList<Particle> results = new ArrayList<Particle>();		// Output.
		double[][] calibration 		= getCalibration(); 				// get calibration table.
		double[] zOffset 			= getZoffset();     // get where z=0 is for each channel in the calibration file.
		double[][] offset 			= getOffset();								// get xyz offset from channel 1.
		double offsetX 				= ij.Prefs.get("SMLocalizer.calibration.Biplane.finalOffsetX",0);
		double offsetY 				= ij.Prefs.get("SMLocalizer.calibration.Biplane.finalOffsetY",0);
		int gWindow 				= (int) ij.Prefs.get("SMLocalizer.calibration.Biplane.window",0);

		for (int i = 0; i < inputResults.size()-1; i++) // loop over all entries
		{
			double searchX = inputResults.get(i).x;			// x coordinate to search for center around.
			double searchY = inputResults.get(i).y;			// y coordinate to search for center around.
			if (inputResults.get(i).include == 1)			// verify that only non processed particles are included.
			{
				inputResults.get(i).include = 2; 			// checked.
				if (inputResults.get(i).x < frameWidth) 	// if on the left hand side.
				{
					/*
					 * check to see if there is another fitted object on the right side within 1 pixel + offset.
					 * if so, use the best fit of these two as dominant and calculate summed intensity within fit region
					 * for both, take ratio and store this. Center coordinate is transposed by offset if the dominant one is on the right.
					 * 
					 */
					searchX += offsetX + frameWidth;		// update search coordinate for second particle.
					searchY += offsetY;						// update search coordinate for second particle.
				}else{
					searchX -= (offsetX + frameWidth);	// update search coordinate for second particle.
					searchY -= offsetY;					// update search coordinate for second particle.
				}
				// find if any object are close to searchX,searchY.
				//System.out.println("searchXY: " + searchX + " : " + searchY + " from:  " + inputResults.get(i).x);
				int j = i + 1;				// start search on next added particle.
				boolean search = true;		// search whilst this is true.

				double z = 0;				// z output.
				int photon = 0;				// photon count.

				while (search)
				{
					if (inputResults.get(j).channel == inputResults.get(i).channel &&  // if the same channel
							inputResults.get(j).frame == inputResults.get(i).frame &&   // and same frame-
							inputResults.get(j).include == 1)							// not previously used.
					{
						if (inputResults.get(j).y < (searchY + inputPixelSize) &&		// if close enough in y direction.
								inputResults.get(j).y > (searchY - inputPixelSize))
						{
							if (inputResults.get(j).x < (searchX + inputPixelSize) &&	// if close enough in x direction.
									inputResults.get(j).x > (searchX - inputPixelSize))
							{											
								inputResults.get(j).include = 2; // this entry has been covered.

								if(inputResults.get(i).x < frameWidth)
								{																							

									double photonLeft 	= getPhotons(image, inputResults.get(i).x/inputPixelSize,inputResults.get(i).y/inputPixelSize, inputResults.get(i).frame, inputResults.get(i).channel, gWindow,totalGain);
									double photonRight 	= getPhotons(image, inputResults.get(j).x/inputPixelSize,inputResults.get(j).y/inputPixelSize, inputResults.get(i).frame, inputResults.get(i).channel, gWindow,totalGain);
									if (photonLeft > 0 && photonRight > 0)
									{
										photon = (int) (photonLeft + photonRight);			
										z = getZ(calibration, zOffset, inputResults.get(i).channel, photonLeft/photonRight);									

									}else
										z = 1E6;
									//System.out.println("i: " + i + "frame: " + inputResults.get(i).frame+ " : " + z + " from " + (photonLeft/photonRight));									

								}else
								{
									double photonLeft 	= getPhotons(image, inputResults.get(j).x/inputPixelSize,inputResults.get(j).y/inputPixelSize, inputResults.get(i).frame, inputResults.get(i).channel, gWindow,totalGain);
									double photonRight 	= getPhotons(image, inputResults.get(i).x/inputPixelSize,inputResults.get(i).y/inputPixelSize, inputResults.get(i).frame, inputResults.get(i).channel, gWindow,totalGain);
									if (photonLeft > 0 && photonRight > 0)
									{
										photon = (int) (photonLeft + photonRight);									
										z = getZ(calibration, zOffset, inputResults.get(i).channel, photonLeft/photonRight);									
									}else
										z = 1E6;
									//	System.out.println("i: " + i + "frame: " + inputResults.get(i).frame+ " : " + z + " from " + (photonLeft/photonRight));

								}
								search = false;
								//								z = 100;

								if (inputResults.get(i).r_square > inputResults.get(j).r_square)	// use the x-y coordinates from the best fit.
								{
									Particle temp 	= new Particle();
									temp.channel 	= inputResults.get(i).channel;
									temp.z 		 	= z;
									temp.frame 	 	= inputResults.get(i).frame;
									temp.photons 	= photon;
									temp.x			= inputResults.get(i).x;
									temp.y			= inputResults.get(i).y;
									temp.sigma_x 	= inputResults.get(i).sigma_x; 		// fitted sigma in x direction.
									temp.sigma_y 	= inputResults.get(i).sigma_y; 		// fitted sigma in y direction.					
									temp.precision_x= inputResults.get(i).precision_x; 	// precision of fit for x coordinate.
									temp.precision_y= inputResults.get(i).precision_y; 	// precision of fit for y coordinate.
									temp.precision_z= 600 / Math.sqrt(temp.photons); 	// precision of fit for z coordinate.
									temp.r_square 	= inputResults.get(i).r_square; 	// Goodness of fit.
									temp.include	= 1; 								// If this particle should be included in analysis and plotted.
									if (inputResults.get(i).x > frameWidth)				// if x coordinate puts this particle on right hand frame, shift it by offset and framewidth.
									{
										temp.x -= (offsetX + frameWidth);
										temp.y -= offsetY;
									}
									if(temp.z != 1E6 && temp.channel>1)	// for all but first channel, shift coordinates to compensate for chromatic shift.
									{
										temp.x -= offset[0][temp.channel-1];
										temp.y -= offset[1][temp.channel-1];
										temp.z -= offset[2][temp.channel-1];
										results.add(temp);
									}

									if (temp.z != 1E6 && temp.channel==1)	// if ok z and first channel.
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
									temp.precision_z= 600 / Math.sqrt(temp.photons); 	// precision of fit for z coordinate.
									temp.r_square 	= inputResults.get(j).r_square; 	// Goodness of fit.
									temp.include	= 1; 								// If this particle should be included in analysis and plotted.
									if (inputResults.get(j).x > frameWidth)				// if x coordinate puts this particle on right hand frame, shift it by offset and framewidth.
									{
										temp.x -= (offsetX + frameWidth);
										temp.y -= offsetY;
									}
									if(temp.z != 1E6 && temp.channel>1)	// for all but first channel, shift coordinates to compensate for chromatic shift.
									{
										temp.x -= offset[0][temp.channel-1];
										temp.y -= offset[1][temp.channel-1];
										temp.z -= offset[2][temp.channel-1];
										results.add(temp);
									}

									if (temp.z != 1E6 && temp.channel==1)	// if ok z and first channel.
										results.add(temp);
								}
							}
						}
					}

					j++;	// test next particle.

					if (j == inputResults.size() || 								// if we're at the end of the list and still have not found any other located events that match.
							inputResults.get(j).frame > inputResults.get(i).frame)	// if we're looking in the wrong frame, stop.
					{
						/*
						 * No corresponding fit at search coordinates was found.
						 */
						search = false;
						if(inputResults.get(i).x < frameWidth) // if particle is on left side.
						{																							
							double photonLeft 	= getPhotons(image, inputResults.get(i).x/inputPixelSize,inputResults.get(i).y/inputPixelSize, inputResults.get(i).frame, inputResults.get(i).channel, gWindow,totalGain);
							double photonRight 	= getPhotons(image, searchX/inputPixelSize,searchY/inputPixelSize, inputResults.get(i).frame, inputResults.get(i).channel, gWindow,totalGain);
							if (photonLeft > 0 && photonRight > 0)
							{
								photon = (int) (photonLeft + photonRight);
								z = getZ(calibration, zOffset, inputResults.get(i).channel, photonLeft/photonRight);																
							}else
								z = 1E6;
							//	System.out.println("i: " + i + "frame: " + inputResults.get(i).frame+ " : " + z + " from " + (photonLeft/photonRight));

						}else 	// if particle is on right side.
						{
							double photonLeft 	= getPhotons(image, searchX/inputPixelSize,searchY/inputPixelSize, inputResults.get(i).frame, inputResults.get(i).channel, gWindow,totalGain);
							double photonRight 	= getPhotons(image, inputResults.get(i).x/inputPixelSize,inputResults.get(i).y/inputPixelSize, inputResults.get(i).frame, inputResults.get(i).channel, gWindow,totalGain);
							if (photonLeft > 0 && photonRight > 0)
							{
								photon = (int) (photonLeft + photonRight);
								z = getZ(calibration, zOffset, inputResults.get(i).channel, photonLeft/photonRight);							
							}else
								z = 1E6;
							//	System.out.println("i: " + i + "frame: " + inputResults.get(i).frame+ " : " + z + " from " + (photonLeft/photonRight));

						}

						//		z=100;
						if (inputResults.get(i).x < frameWidth)				// if particle is on the left side, no need to shift by framewidth.
						{
							Particle temp 	= new Particle();
							temp.channel 	= inputResults.get(i).channel;
							temp.z 		 	= z;
							temp.frame 	 	= inputResults.get(i).frame;
							temp.photons 	= photon;
							temp.x			= inputResults.get(i).x;
							temp.y			= inputResults.get(i).y;
							temp.sigma_x 	= inputResults.get(i).sigma_x; 		// fitted sigma in x direction.
							temp.sigma_y 	= inputResults.get(i).sigma_y; 		// fitted sigma in y direction.					
							temp.precision_x= inputResults.get(i).precision_x; 	// precision of fit for x coordinate.
							temp.precision_y= inputResults.get(i).precision_y; 	// precision of fit for y coordinate.
							temp.precision_z= 600 / Math.sqrt(temp.photons); 	// precision of fit for z coordinate.
							temp.r_square 	= inputResults.get(i).r_square; 	// Goodness of fit.
							temp.include	= 1; 								// If this particle should be included in analysis and plotted.
							if(temp.z != 1E6 && temp.channel>1)	// for all but first channel, shift coordinates to compensate for chromatic shift.
							{
								temp.x -= offset[0][temp.channel-1];
								temp.y -= offset[1][temp.channel-1];
								temp.z -= offset[2][temp.channel-1];
								results.add(temp);
							}

							if (temp.z != 1E6 && temp.channel==1)	// if ok z and first channel.
								results.add(temp);
						}else
						{
							Particle temp 	= new Particle();
							temp.channel 	= inputResults.get(i).channel;
							temp.z 		 	= z;
							temp.frame 	 	= inputResults.get(i).frame;
							temp.photons 	= photon;
							temp.x			= inputResults.get(i).x - frameWidth - offsetX;	// shift to left side.
							temp.y			= inputResults.get(i).y - offsetY;
							temp.sigma_x 	= inputResults.get(i).sigma_x; 		// fitted sigma in x direction.
							temp.sigma_y 	= inputResults.get(i).sigma_y; 		// fitted sigma in y direction.					
							temp.precision_x= inputResults.get(i).precision_x; 	// precision of fit for x coordinate.
							temp.precision_y= inputResults.get(i).precision_y; 	// precision of fit for y coordinate.
							temp.precision_z= 600 / Math.sqrt(temp.photons); 	// precision of fit for z coordinate.
							temp.r_square 	= inputResults.get(i).r_square; 	// Goodness of fit.
							temp.include	= 1; 								// If this particle should be included in analysis and plotted.
							if(temp.z != 1E6 && temp.channel>1)	// for all but first channel, shift coordinates to compensate for chromatic shift.
							{
								temp.x -= offset[0][temp.channel-1];
								temp.y -= offset[1][temp.channel-1];
								temp.z -= offset[2][temp.channel-1];
								results.add(temp);
							}

							if (temp.z != 1E6 && temp.channel==1)	// if ok z and first channel.
								results.add(temp);
						}
					}
				}
			}
		}		
		results = shiftXY(results);
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
		boolean[][] include 	= new boolean[7][10];							// for filtering of fitted results.
		double[][] lb 			= new double[7][10];							// for filtering of fitted results.
		double[][] ub 			= new double[7][10];							// for filtering of fitted results.
		for (int i = 0; i < 10; i++)											// populate "include", "lb" and "ub".
		{
			// logical list of which parameters to filter against.
			include[0][i] 		= false;
			include[1][i] 		= false;
			include[2][i] 		= true;		// r_square.
			include[3][i] 		= false;
			include[4][i] 		= false;
			include[5][i] 		= false;
			include[6][i] 		= false;
			// lower bounds.
			lb[0][i]			= 0;
			lb[1][i]			= 0;
			lb[2][i]			= 0.8;		// r_square.
			lb[3][i]			= 0;
			lb[4][i]			= 0;
			lb[5][i]			= 0;
			lb[6][i]			= 0;
			// upper bounds.
			ub[0][i]			= 0;
			ub[1][i]			= 0;
			ub[2][i]			= 1.0;		// r_square.
			ub[3][i]			= 0;
			ub[4][i]			= 0;
			ub[5][i]			= 0;
			ub[6][i]			= 0;

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
				float[] median = new float[IP.getPixelCount()];
				for (int i = 0; i < IP.getPixelCount(); i++)
					median[i] = IP.getf(i);
				BackgroundCorrection.quickSort(median, 0, median.length-1);
				int value = 0;
				for (int i = 0; i < IP.getPixelCount(); i++)
				{								
					value = (int)(IP.get(i)-median[(int)(median.length/2)]);
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
		int gWindow 		= 3;
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
			gWindow = gWindow + 2; // increase window size each loop.
			for (double level = 0.7; level > 0.3; level -= 0.1)
			{
				for (double maxSigma = 2.0; maxSigma < 4; maxSigma += 0.5)
				{					
					int[] MinLevel 			= new int[10]; // precast.
					for (int ch = 1; ch <= nChannels; ch++) // set level in a channel by channel specific manner.
					{
						if (image.getNFrames() == 1)
						{
							image.setPosition(							
									ch,			// channel.
									(int) nFrames/2,			// slice.
									1);		// frame.
						}
						else
						{														
							image.setPosition(
									ch,			// channel.
									1,			// slice.
									(int) nFrames/2);		// frame.
						}
						ImageStatistics IMstat 	= image.getStatistics(); 					
						MinLevel[ch-1] = (int)(IMstat.max*level);
					}

					Fit3D.fit(MinLevel,gWindow,inputPixelSize,totalGain,maxSigma);
					//					localizeAndFit.run(MinLevel, gWindow, inputPixelSize, totalGain, 0, maxSigma, "2D");
					/*
					 * clean out fits based on goodness of fit:
					 */

					cleanParticleList.run(lb,ub,include);
					cleanParticleList.delete();
					ArrayList<Particle> result = TableIO.Load();
					for (int i = 0; i < result.size(); i++)				// loop over all entries.
					{
						result.get(i).z = (result.get(i).frame-1)*zStep; // set z to zStep in nm * frame index.
					}
					TableIO.Store(result);
					result 				= TableIO.Load();
					double[][] ratio	= new double[nFrames][nChannels];
					int[][] count		= new int[nFrames][nChannels];
					/*
					 * Find offset based on center stack fits.
					 */

					double offsetX 	= 0;	// used to find offset between left and right hand side of frame, excluding width of frame.
					double offsetY 	= 0;	// used to find offset between left and right hand side of frame, excluding width of frame.
					int offsetCount = 0;	// counter.


					for (int i = 0; i < result.size(); i++)	// loop over all entries.
					{
						if (result.get(i).frame > ((int)nFrames/2-5) &&	// if we're on the within the 5 slices closest to the center
								result.get(i).frame < ((int)nFrames/2+5) &&
								result.get(i).x < frameWidth) 			// and on the left side of the image.
						{
							/*
							 * find closest fit on right hand side of the image from that frame, frameWidth to the right.
							 */
							double tempOffset 	= frameWidth;	// guess for actual distance.
							double tempOffsetX 	= frameWidth;	// guess for offset in x.
							double tempOffsetY 	= frameWidth;	// guess for offset in y.
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
								searchX -= (offsetX + frameWidth);
								searchY -= offsetY;
							}

							// find if any object are close to searchX,searchY.
							int j = i + 1;
							boolean search = true;
							while (search)
							{
								if (result.get(j).channel == result.get(i).channel &&	// if the same channel
										result.get(j).frame == result.get(i).frame &&   // and same frame-
										result.get(j).include == 1)						// not previously included.
								{
									if (result.get(j).y < (searchY + inputPixelSize) &&
											result.get(j).y > (searchY - inputPixelSize))
									{
										if (result.get(j).x < (searchX + inputPixelSize) &&
												result.get(j).x > (searchX - inputPixelSize))
										{											
											result.get(j).include = 2; // this entry has been covered.
											if(result.get(i).x < frameWidth)
											{																							
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


					int minLength = 40;										// minimum length of calibration range.				
					double[][] calibration = makeCalibrationCurve(ratio, minLength, nChannels,false,false,false);// create calibration curve.

					if (calibrationLength < calibration.length)				// if the new calibration using current parameter settings covers a larger range.
					{
						calibrationLength = calibration.length;				// update best z range.
						meanRsquare = 0;									// set quailty check to 0.						
					}
					if (calibrationLength == calibration.length) 			// if of equal length (or if it was just updated).
					{
						double rsquare = 0;
						for (int i = 0; i < result.size(); i++)
						{
							rsquare += result.get(i).r_square;
						}
						rsquare /= result.size();
						if (rsquare > meanRsquare)							// if the new parameters yield better average fit.
						{								
							meanRsquare = rsquare;							// update
							finalLevel = level;								// update
							finalSigma = maxSigma;							// update
							finalGWindow = gWindow;							// update
							finalOffsetX = offsetX;							// update
							finalOffsetY = offsetY;							// update
						}
					}			
				} // iterate over maxSigma
			} // iterate over level.
			loopC++;

		}
		/*
		 * Create calibration curve based on optimal parameters iterated over in the above code.
		 */

		int[] MinLevel 			= new int[10]; // precast.
		for (int ch = 1; ch <= nChannels; ch++) // set level in a channel by channel specific manner.
		{
			if (image.getNFrames() == 1)
			{
				image.setPosition(							
						ch,			// channel.
						(int) nFrames/2,			// slice.
						1);		// frame.
			}
			else
			{														
				image.setPosition(
						ch,			// channel.
						1,			// slice.
						(int) nFrames/2);		// frame.
			}
			ImageStatistics IMstat 	= image.getStatistics(); 					
			MinLevel[ch-1] = (int)(IMstat.max*finalLevel);
		}
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
					searchX -= (finalOffsetX + frameWidth);
					searchY -= finalOffsetY;
				}
				// find if any object are close to searchX,searchY.
				int j = i + 1;
				boolean search = true;
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

								if(result.get(i).x < frameWidth)
								{																							
									double photonLeft 	= getPhotons(image, result.get(i).x/inputPixelSize,result.get(i).y/inputPixelSize, result.get(i).frame, result.get(i).channel, finalGWindow,totalGain);
									double photonRight 	= getPhotons(image, result.get(j).x/inputPixelSize,result.get(j).y/inputPixelSize, result.get(i).frame, result.get(i).channel, finalGWindow,totalGain);												
									ratio[result.get(i).frame][result.get(i).channel-1] += photonLeft/photonRight;
									count[result.get(i).frame][result.get(i).channel-1]++;					

								}else
								{
									double photonLeft 	= getPhotons(image, result.get(j).x/inputPixelSize,result.get(j).y/inputPixelSize, result.get(i).frame, result.get(i).channel, finalGWindow,totalGain);
									double photonRight 	= getPhotons(image, result.get(i).x/inputPixelSize,result.get(i).y/inputPixelSize, result.get(i).frame, result.get(i).channel, finalGWindow,totalGain);												
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
							double photonLeft 	= getPhotons(image, result.get(i).x/inputPixelSize,result.get(i).y/inputPixelSize, result.get(i).frame, result.get(i).channel, finalGWindow,totalGain);
							double photonRight 	= getPhotons(image, searchX/inputPixelSize,searchY/inputPixelSize, result.get(i).frame, result.get(i).channel, finalGWindow,totalGain);												
							ratio[result.get(i).frame][result.get(i).channel-1] += photonLeft/photonRight;
							count[result.get(i).frame][result.get(i).channel-1]++;						

						}else
						{
							double photonLeft 	= getPhotons(image, searchX/inputPixelSize,searchY/inputPixelSize, result.get(i).frame, result.get(i).channel, finalGWindow,totalGain);
							double photonRight 	= getPhotons(image, result.get(i).x/inputPixelSize,result.get(i).y/inputPixelSize, result.get(i).frame, result.get(i).channel, finalGWindow,totalGain);												
							ratio[result.get(i).frame][result.get(i).channel-1] += photonLeft/photonRight;
							count[result.get(i).frame][result.get(i).channel-1]++;											
						}
					}
				}

			}
		}


		//double[] print = new double[count.length];
		for (int Ch = 1; Ch <= nChannels; Ch++)
		{
			for(int i = 0; i < count.length; i++)
			{
				if (count[i][Ch-1]>0)
				{
					ratio[i][Ch-1] 	/= count[i][Ch-1]; // mean ratio for this z-depth.
					//					print[i] = ratio[i][Ch-1];
				}
			}
		}
		int minLength = 40;			
		/*	
		Plot plot2 = new Plot("Biplane calibration", "z [nm]", "Ratio");

double[] x2 = new double[print.length];
for (int i =0; i < print.length; i++)
	x2[i]= i;

			plot2.addPoints(x2,print, Plot.LINE);

		plot2.show();
		 */
		double[][] calibration = makeCalibrationCurve(ratio, minLength, nChannels,false,false,true);

		/*
		 * STORE calibration file:
		 */
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
		for (int ch = 1; ch <= nChannels; ch++)
		{
			for (int i = 0; i < calibration.length; i++)
			{
				ij.Prefs.set("SMLocalizer.calibration.Biplane.xOffset.Ch"+ch+"."+i,0); // reset.
				ij.Prefs.set("SMLocalizer.calibration.Biplane.yOffset.Ch"+ch+"."+i,0); // reset.
			}
		}
		ij.Prefs.savePreferences(); // store settings.
		for (int idx = 0; idx < result.size(); idx++)
		{
			result.get(idx).include = 1;
		}
		ArrayList<Particle> resultCalib = fit(result,inputPixelSize, totalGain);
		TableIO.Store(resultCalib);
		cleanParticleList.run(lb,ub,include);
		cleanParticleList.delete();
		resultCalib = TableIO.Load();
		double[][][] xyOffset = getXYoffset(resultCalib,calibration,nFrames);				
		//		System.out.println(finalOffsetY + " : " + finalOffsetX);
		ij.measure.ResultsTable tab = Analyzer.getResultsTable();
		tab.reset();
		tab.show("Results");
		/*
		 * Go through central part of the fit for each channel and calculate offset in XY for each channel.
		 */

		for (int ch = 1; ch <= nChannels; ch++)
		{
			for (int i = 0; i < calibration.length; i++)
			{
				ij.Prefs.set("SMLocalizer.calibration.Biplane.xOffset.Ch"+ch+"."+i,xyOffset[0][i][ch-1]);
				ij.Prefs.set("SMLocalizer.calibration.Biplane.yOffset.Ch"+ch+"."+i,xyOffset[1][i][ch-1]);
			}
		}
		ij.Prefs.savePreferences(); // store settings.

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
				ij.Prefs.set("SMLocalizer.calibration.Biplane.ChOffsetX"+i,offset[0][i-1]);
				ij.Prefs.set("SMLocalizer.calibration.Biplane.ChOffsetY"+i,offset[1][i-1]);
				ij.Prefs.set("SMLocalizer.calibration.Biplane.ChOffsetZ"+i,offset[2][i-1]);
			}
			ij.Prefs.savePreferences(); // store settings.
		}else
		{
			for (int i = 1; i < 10; i++)
			{
				ij.Prefs.set("SMLocalizer.calibration.Biplane.ChOffsetX"+i,0);
				ij.Prefs.set("SMLocalizer.calibration.Biplane.ChOffsetY"+i,0);
				ij.Prefs.set("SMLocalizer.calibration.Biplane.ChOffsetZ"+i,0);
			}
			ij.Prefs.savePreferences(); // store settings.
		}
		Plot plot = new Plot("Biplane calibration", "z [nm]", "Ratio");
		double[] zOffset = getZoffset();
		for (int ch = 0; ch < calibration[0].length; ch++)
		{
			double[] printout = new double[calibration.length];
			double[] x = new double[printout.length];
			for (int i = 0; i < printout.length; i++)
			{
				printout[i] = calibration[i][ch];
				x[i] = (i-zOffset[ch])*zStep;
			}
			if (ch == 0)
				plot.setColor(Color.BLACK);
			if (ch == 1)
				plot.setColor(Color.BLUE);
			if (ch == 2)
				plot.setColor(Color.RED);
			if (ch == 3)
				plot.setColor(Color.GREEN);
			if (ch == 4)
				plot.setColor(Color.MAGENTA);
			plot.addPoints(x,printout, Plot.LINE);
		}
		if (calibration[0].length == 1)
			plot.addLegend("Ch 1 \n Ch 2");
		if (calibration[0].length == 2)
			plot.addLegend("Ch 1 \n Ch 2 \n Ch 3");		
		if (calibration[0].length == 3)
			plot.addLegend("Ch 1 \n Ch 2 \n Ch 3 \n Ch 4");
		if (calibration[0].length == 4)
			plot.addLegend("Ch 1 \n Ch 2 \n Ch 3 \n Ch 4 \n Ch 5");
		plot.show();

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
		if (x >= 0 && y >= 0){
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
		}
		return photons;
	}
	public static double[][] getOffset()
	{
		double[][] offset = new double[3][(int) ij.Prefs.get("SMLocalizer.calibration.Biplane.channels",0)];
		for (int i = 1; i < offset[0].length; i++)
		{
			offset[0][i-1] = ij.Prefs.get("SMLocalizer.calibration.Biplane.ChOffsetX"+i,0);
			offset[1][i-1] = ij.Prefs.get("SMLocalizer.calibration.Biplane.ChOffsetY"+i,0);
			offset[2][i-1] = ij.Prefs.get("SMLocalizer.calibration.Biplane.ChOffsetZ"+i,0);
		}

		return offset;
	}
	public static double[][][] getXYoffset(ArrayList<Particle> inputParticle,double[][] calibration, int nFrames)
	{
		double[][][] xyOffset 	= new double[2][calibration.length][calibration[0].length]; // x-y (z,ch). 
		double[] center 		= getZoffset();			// get info of where z=0 is in the calibration file.
		nFrames = Math.round(nFrames/2); // center frame.
		//System.out.println("center frame: " +nFrames + " vs "  + center[0] + " : "+calibration.length + " start at " + (nFrames - center[0]));
		for (int ch = 0; ch < calibration[0].length; ch++)	// over all channels.
		{
			int chStart 	= -1;
			int nCenter 	= 0; // number of locations to search after.
			for (int idx 	= 0; idx < inputParticle.size(); idx++)
			{				
				if (inputParticle.get(idx).channel == ch+1 && inputParticle.get(idx).frame >= (nFrames - center[ch]) && chStart == -1)
					chStart = idx;
				if (inputParticle.get(idx).frame == nFrames && inputParticle.get(idx).channel == ch+1) // if we're at the frame corresponding to z = 0 and in the correct channel.
					//					if (inputParticle.get(idx).frame == center[ch] && inputParticle.get(idx).channel == ch+1) // if we're at the frame corresponding to z = 0 and in the correct channel.
				{
					nCenter++;
				}
			}
			if (nCenter > 0)	// if we found centers.
			{				
				double[][] localZeroXY = new double[2][nCenter];	// array to store all individual offsets.
				int counter  = 0;	
				for (int idx = 0; idx < inputParticle.size(); idx++)
				{
					if (inputParticle.get(idx).frame == nFrames && inputParticle.get(idx).channel == ch+1)	// if we're at the center frame.
						//						if (inputParticle.get(idx).frame == center[ch] && inputParticle.get(idx).channel == ch+1)	// if we're at the center frame.
					{
						localZeroXY[0][counter] = inputParticle.get(idx).x;	// store x coordinate.
						localZeroXY[1][counter] = inputParticle.get(idx).y;	// store y coordinate.
						counter++;
					}
				}// localZeroXY now contains x-y coordinates for central slice particles.
				/*
				 * Find smallest offset against localZeroXY for each particle in stack for this channel. Take mean, remove outliers.
				 */
				int startIdx = chStart; // which inded to start from.
				while (inputParticle.get(startIdx).channel != ch+1) // jump forward to first index for the current channel,
					startIdx ++;
				int endIdx		 = startIdx; // final index for this frame and channel.
				boolean optimize = true;	// loop whilst this is true.

				while (optimize)
				{
					counter = 0;	// keep track of number of added corrections.
					endIdx = startIdx;	// restart.
					boolean loop = true;	// exit flag for final frame and channel.
					if (inputParticle.size() > endIdx+1) // if we're not at the end of the list.
					{
						while (inputParticle.get(startIdx).frame == inputParticle.get(endIdx).frame && inputParticle.get(endIdx).channel == ch+1 && loop) // as long as we're within the same frame and channel.
						{
							if (inputParticle.size() > endIdx+2) // if we're not at the end
							{
								endIdx++;						 // step forward

							}else
								loop = false; // exit for final frame and channel.
						}
					}
					double[][] frameOffset = new double[2][endIdx-startIdx + 1]; // preallocate
					double meanX = 0;	// mean x offset.
					double meanY = 0;	// mean y offset.
					while (startIdx <= endIdx) // whilst we're not done with this frame.
					{
						double z = inputParticle.get(startIdx).z / ij.Prefs.get("SMLocalizer.calibration.Biplane.step",0);  // get back approximate frame.
						//	z += center[ch];																						// shift by center to get frame number.
						z += nFrames;																						// shift by center to get frame number.
						//		System.out.println(inputParticle.get(startIdx).frame + " vs  " + z   ); 
						if (z > (inputParticle.get(startIdx).frame - 4) && z < (inputParticle.get(startIdx).frame + 4)) 		// if we're within +/- 3 frames of the current one (ie if the fit worked with 3*zStep precision)
						{
							double distance = 500; // max distance to be considered in nm.
							for (int i = 0; i < localZeroXY[0].length; i ++) // loop over all particles from center frame.
							{
								double tempDistance = Math.sqrt((inputParticle.get(startIdx).x - localZeroXY[0][i])*(inputParticle.get(startIdx).x - localZeroXY[0][i]) + // calculate distance from current particle to the current center frame particle.
										(inputParticle.get(startIdx).y - localZeroXY[1][i])*(inputParticle.get(startIdx).y - localZeroXY[1][i]));
								if (tempDistance < distance) // if the particles are closer then the previous loops (or start value)
								{
									frameOffset[0][counter] = localZeroXY[0][i] - inputParticle.get(startIdx).x;	// update offset for this particle
									frameOffset[1][counter] = localZeroXY[1][i] - inputParticle.get(startIdx).y;	// update offset for this particle
									distance = tempDistance;														// update distance.
								}
							} // nearest neighbor found.
							meanX += frameOffset[0][counter]; 	// add offset in x to mean.
							meanY += frameOffset[1][counter]; 	// add offset in y to mean.
							counter ++;						 	// step forward in the array
						}
						startIdx++;								// next particle within the frame.
					}
					double sigmaX = 0;	// std of x offsets.
					double sigmaY = 0; 	// std of y offsets.
					if (counter > 0)	// if we've added values (ie found particles close enough to the center frame particles)
					{
						if (counter > 1) // sample requires more then 1 entry.
						{
							for (int i = 0; i < frameOffset[0].length; i++)	// loop over all offsets in this frame.
							{
								sigmaX += ((frameOffset[0][i] - meanX/counter)*(frameOffset[0][i] - meanX/counter))/(counter-1);	// add (Xi-µ)/(n-1)
								sigmaY += ((frameOffset[1][i] - meanY/counter)*(frameOffset[1][i] - meanY/counter))/(counter-1);	// add (Yi-µ)/(n-1)
							}
							sigmaX = Math.sqrt(sigmaX);	// square root of sigmax^2.
							sigmaY = Math.sqrt(sigmaY);	// square root of sigmay^2.	
							//			System.out.println("frame: " + inputParticle.get(endIdx).frame + " mean" + meanX/counter + " x " + meanY/counter + " sigma: " + sigmaX + " x " + sigmaY);
							for (int i = 0; i < frameOffset[0].length; i++)	// loop over all offsets in this frame.
							{	
								//			System.out.println(frameOffset[0][i] + " x " + frameOffset[1][i]);
								if ((frameOffset[0][i]) < (meanX/counter - sigmaX*3) &&			// if we're not within 3 sigma away from mean.
										(frameOffset[0][i]) > (meanX/counter + sigmaX*3) &&
										(frameOffset[1][i]) < (meanY/counter - sigmaY*3) &&
										(frameOffset[1][i]) > (meanY/counter + sigmaY*3))
								{
									meanX -= frameOffset[0][i];		// remove this entry
									meanY -= frameOffset[1][i];		// remove this entry
									counter--;						// decrease.
								}
							}
						}
						if (counter > 0)
						{
							meanX /= counter;	// calculate mean.
							meanY /= counter;	// calculate mean
							xyOffset[0][inputParticle.get(startIdx).frame - inputParticle.get(chStart).frame][ch] = meanX; // remove this value from x for this z (or rather frame) and channel.
							xyOffset[1][inputParticle.get(startIdx).frame - inputParticle.get(chStart).frame][ch] = meanY; // remove this value from x for this z (or rather frame) and channel.
						}
					}
					//			System.out.println(idxCounter + " : "+ xyOffset[0][idxCounter][ch] + " x " + xyOffset[1][idxCounter][ch] + " : " + startIdx);
					startIdx++;	// step to the next frame.
					if (startIdx >= inputParticle.size())	// if we're done.
						optimize = false;
					else if ((inputParticle.get(startIdx).frame - inputParticle.get(chStart).frame) >= calibration.length-1)
						optimize = false;
					else if (inputParticle.get(startIdx).channel != ch + 1)	// if we've changed channel.
						optimize = false;					
				} // while(optimize)

			} // if there are any centers in the central slice.
		}
		for (int idx =0; idx < calibration.length; idx++)
		{
			//	System.out.println(idx + " : " + xyOffset[0][idx][0] + " x " + xyOffset[1][idx][0]);
		}
		/*
		 * interpolate
		 */
		for (int ch = 0; ch <  calibration[0].length; ch++)
		{
			for (int XY = 0; XY < 2; XY++)
			{
				for (int idx = 0; idx < calibration.length; idx++)
				{
					if (xyOffset[XY][idx][ch] == 0 && idx != nFrames)
					{
						if (idx == 0) // if the first entry is 0
						{
							int tempIdx = idx+1;
							int added = 0;
							double total = 0;
							while (added < 2 && tempIdx != calibration.length)
							{
								if (xyOffset[XY][tempIdx][ch] != 0)
								{
									added++;
									if (added == 1)
										total -= xyOffset[XY][tempIdx][ch];
									else
										total += xyOffset[XY][tempIdx][ch];
									//	System.out.println("adding " + xyOffset[XY][tempIdx][ch] + " from " + tempIdx);
								}
								tempIdx++;
							}

							tempIdx--;
							//System.out.println("corr: " + (tempIdx-idx)+ " total: " + total );
							total /= (tempIdx-idx);

							while (tempIdx >= idx)
							{
								if (xyOffset[XY][tempIdx][ch] == 0)
								{
									xyOffset[XY][tempIdx][ch] = xyOffset[XY][tempIdx+1][ch] - total;	
									//System.out.println(xyOffset[XY][tempIdx][ch] + " from " + xyOffset[XY][tempIdx+1][ch] + " - " + total);

								}
								tempIdx--;
							}
						}else if (idx == calibration.length-1)
						{
							double total = xyOffset[XY][idx-1][ch] - xyOffset[XY][idx-2][ch];														
							xyOffset[XY][idx][ch] = xyOffset[XY][idx-1][ch] + total;
						}
						else // central part.
						{
							double total = -xyOffset[XY][idx-1][ch];
							int added = 1;
							int tempIdx = idx + 1;
							while (added < 2 && tempIdx != calibration.length)
							{
								if (xyOffset[XY][tempIdx][ch] != 0)
								{
									added++;
									total += xyOffset[XY][tempIdx][ch];								
								}
								tempIdx++;
							}
							if (tempIdx != calibration.length)
							{
								tempIdx--;
								total /= (tempIdx-idx);
								while (tempIdx >= idx)
								{
									if (xyOffset[XY][tempIdx][ch] == 0)
									{
										xyOffset[XY][tempIdx][ch] = xyOffset[XY][tempIdx+1][ch] - total;	
									}
									tempIdx--;
								}
							}else if (added == 1) // if we do not have a value at the end.
							{
								total = xyOffset[XY][idx-1][ch] - xyOffset[XY][idx-2][ch];
								tempIdx = idx + 1;
								while (tempIdx < calibration.length)
								{
									xyOffset[XY][tempIdx][ch] = xyOffset[XY][tempIdx-1][ch] + total;
									tempIdx++;
								}
							}
						}
					}
				}
			}
		}
		for (int idx =0; idx < calibration.length; idx++)
		{
			//		System.out.println(idx + " : " + xyOffset[0][idx][0] + " x " + xyOffset[1][idx][0]);
		}


		/*
		 * 5 point mean filtering.
		 */
		for (int ch = 0; ch < calibration[0].length; ch++)
		{
			double[] tempVector = new double[xyOffset[0].length];	// 5 point smoothed data.
			for (int XY = 0; XY < 2; XY++)
			{
				int idx = 0;										// index variable.				

				while (idx < tempVector.length)
				{
					// 5 point smoothing:
					if (idx == 0)
					{
						int included = 0;
						if (xyOffset[XY][idx][ch] != 0)
							included++;
						if (xyOffset[XY][idx + 1][ch] != 0)
							included++;
						if (xyOffset[XY][idx + 2][ch] != 0)
							included++;
						if (included > 0)
						{
							tempVector[idx] = (xyOffset[XY][idx][ch]
									+ xyOffset[XY][idx + 1][ch] 
											+ xyOffset[XY][idx + 2][ch])/included;
						}
						else
						{
							tempVector[idx] = 0;
						}
					}else if (idx == 1)
					{
						int included = 0;					
						if (xyOffset[XY][idx - 1][ch] != 0)
							included++;
						if (xyOffset[XY][idx][ch] != 0)
							included++;
						if (xyOffset[XY][idx + 1][ch] != 0)
							included++;
						if (xyOffset[XY][idx + 2][ch] != 0)
							included++;
						if (included > 0)
						{
							tempVector[idx] = (xyOffset[XY][idx - 1][ch]
									+ xyOffset[XY][idx][ch]
											+ xyOffset[XY][idx + 1][ch] 
													+ xyOffset[XY][idx + 2][ch])/included;
						}
						else
						{
							tempVector[idx] = 0;

						}
					}else if (idx == xyOffset[XY].length - 2)
					{
						int included = 0;
						if (xyOffset[XY][idx - 2][ch] != 0)
							included++;
						if (xyOffset[XY][idx - 1][ch] != 0)
							included++;
						if (xyOffset[XY][idx][ch] != 0)
							included++;
						if (xyOffset[XY][idx + 1][ch] != 0)
							included++;
						if (included > 0)
						{
							tempVector[idx] = (xyOffset[XY][idx - 2][ch]
									+ xyOffset[XY][idx - 1][ch]
											+ xyOffset[XY][idx][ch] 
													+ xyOffset[XY][idx + 1][ch])/included;
						}
						else
						{
							tempVector[idx] = 0;

						}
					}else if (idx == xyOffset[XY].length - 1)
					{
						int included = 0;
						if (xyOffset[XY][idx - 2][ch] != 0)
							included++;
						if (xyOffset[XY][idx - 1][ch] != 0)
							included++;
						if (xyOffset[XY][idx][ch] != 0)
							included++;
						if (included > 0)
						{
							tempVector[idx] = (xyOffset[XY][idx - 2][ch]
									+ xyOffset[XY][idx - 1][ch]
											+ xyOffset[XY][idx][ch])/included;
						}
						else
						{
							tempVector[idx] = 0;

						}
					}else if (idx < xyOffset[XY].length - 2)
					{
						int included = 0;
						if (xyOffset[XY][idx - 2][ch] != 0)
							included++;
						if (xyOffset[XY][idx - 1][ch] != 0)
							included++;
						if (xyOffset[XY][idx][ch] != 0)
							included++;
						if (xyOffset[XY][idx + 1][ch] != 0)
							included++;
						if (xyOffset[XY][idx + 2][ch] != 0)
							included++;
						if (included > 0)
						{
							tempVector[idx] = (xyOffset[XY][idx - 2][ch]
									+ xyOffset[XY][idx - 1][ch]
											+ xyOffset[XY][idx][ch] 
													+ xyOffset[XY][idx + 1][ch]
															+ xyOffset[XY][idx + 2][ch])/included;
						}
						else
						{
							tempVector[idx] = 0;

						}
					}
					idx++;

				} // tempVector populated.
				for (int i = 0; i < tempVector.length; i++ ) // Transfer temp variable back to main vector.
					xyOffset[XY][i][ch] = tempVector[i];
			}// loop over x and y


		}// channel loop.

		/*
		Plot plot = new Plot("PRILM calibration", "z [nm]", "Angle [rad]");
		double[] zOffset = getZoffset();
		for (int ch = 0; ch < 2; ch++)
		{
			double[] printout = new double[calibration.length];
			double[] x = new double[printout.length];
			for (int i = 0; i < printout.length; i++)
			{
				printout[i] = xyOffset[ch][i][0];
				x[i] = (i-zOffset[0]);
			}
			if (ch == 0)
				plot.setColor(Color.BLACK);
			if (ch == 1)
				plot.setColor(Color.BLUE);
			if (ch == 2)
				plot.setColor(Color.RED);
			if (ch == 3)
				plot.setColor(Color.GREEN);
			if (ch == 4)
				plot.setColor(Color.MAGENTA);
			plot.addPoints(x,printout, Plot.LINE);
		}
		if (calibration[0].length == 1)
			plot.addLegend("Ch 1 \n Ch 2");
		if (calibration[0].length == 2)
			plot.addLegend("Ch 1 \n Ch 2 \n Ch 3");		
		if (calibration[0].length == 3)
			plot.addLegend("Ch 1 \n Ch 2 \n Ch 3 \n Ch 4");
		if (calibration[0].length == 4)
			plot.addLegend("Ch 1 \n Ch 2 \n Ch 3 \n Ch 4 \n Ch 5");
		plot.show();
		/*	for (int i = 0; i < calibration.length; i++)
		{
			System.out.println("idx: " + i + " offset: " + xyOffset[0][i][0] + " x " +xyOffset[1][i][0]);
		}
		 */
		return xyOffset;
	}

	public static ArrayList<Particle> shiftXY(ArrayList<Particle> inputList)
	{
		double[][][] xyOffset = new double[2][(int) ij.Prefs.get("SMLocalizer.calibration.Biplane.height",0)][(int) ij.Prefs.get("SMLocalizer.calibration.Biplane.channels",0)]; // precast.
		for(int ch = 1; ch <= xyOffset[0][0].length; ch++) // load in correction table.
		{
			for (int i = 0; i < xyOffset[0].length; i++)	// loop over all z positions.

			{
				xyOffset[0][i][ch-1] =  ij.Prefs.get("SMLocalizer.calibration.Biplane.xOffset.Ch"+ch+"."+i,0);
				xyOffset[1][i][ch-1] =  ij.Prefs.get("SMLocalizer.calibration.Biplane.yOffset.Ch"+ch+"."+i,0);
			}
		}	
		double zStep =  ij.Prefs.get("SMLocalizer.calibration.Biplane.step",0);
		double[] center = getZoffset();
		for (int idx = 0; idx < inputList.size(); idx++)		 // loop over all particles.
		{
			double z = inputList.get(idx).z /zStep;				// normalize z position based on calibration z-step size.
			z += center[inputList.get(idx).channel-1];			// shift z by center to get back approximate frame number from stack. Use this to know what value to use from the offset table.
			double shiftX = 0;
			double shiftY = 0;
			if (z >= 0 && z < xyOffset[0].length)				// if within ok range.
			{
				shiftX = xyOffset[0][(int)Math.floor(z)][inputList.get(idx).channel-1];
				shiftX += (xyOffset[0][(int)(Math.floor(z)+1)][inputList.get(idx).channel-1] -xyOffset[0][(int)z][inputList.get(idx).channel-1]) * (Math.floor(z+1)-(int)(Math.floor(z)));
				shiftY = xyOffset[1][(int)Math.floor(z)][inputList.get(idx).channel-1];
				shiftY += (xyOffset[1][(int)(Math.floor(z)+1)][inputList.get(idx).channel-1] -xyOffset[1][(int)z][inputList.get(idx).channel-1]) * (Math.floor(z+1)-(int)(Math.floor(z)));
			}else if ((int)z == xyOffset[0].length) // special case if we're at the end of the table.
			{
				shiftX = xyOffset[0][(int)Math.floor(z)][inputList.get(idx).channel-1];
				shiftY = xyOffset[1][(int)Math.floor(z)][inputList.get(idx).channel-1];			
			}
			inputList.get(idx).x +=shiftX;	// add the compensation to x.
			inputList.get(idx).y +=shiftY;  // add the compensation to y.
		}

		return inputList;	// return corrected result.
	}
	public static double[] getZoffset()
	{
		int nChannels = (int)ij.Prefs.get("SMLocalizer.calibration.Biplane.channels",0);
		double[] zOffset = new double[nChannels];
		for (int Ch = 1; Ch <= nChannels; Ch++)
		{		
			zOffset[Ch-1] = ij.Prefs.get("SMLocalizer.calibration.Biplane.center.Ch"+Ch,0);
		} 	
		return zOffset;
	}
	public static double getZ (double[][] calibration, double[] zOffset, int channel, double ratio)
	{
		//System.out.println(ratio + " from " + calibration[0][channel-1] + " to "+ calibration[calibration.length-1][channel-1]);
		double z = 0;
		int idx = 1;
		while (idx < calibration.length - 1 && calibration[idx][channel-1] > ratio)
		{
			idx++;
		}
		if (idx == calibration.length -1 && ratio < calibration[idx][channel-1])		
			z = 1E6;
		else if (calibration[idx][channel-1] == ratio)
			z = idx;
		else if (calibration[0][channel-1] == ratio)
			z = 0;
		else if (calibration[0][channel-1] < ratio)
			z = 1E6;
		else // interpolate
		{
			double diff = calibration[idx-1][channel-1] - calibration[idx][channel-1];
			double fraction = (ratio - calibration[idx][channel-1]) / diff;
			z = idx - 1 + fraction;
		} 					

		if (z != 1E6)
		{
			z -= zOffset[channel-1];
			z *= ij.Prefs.get("SMLocalizer.calibration.Biplane.step",0); // scale.
		}
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
	public static double[][] makeCalibrationCurve(double[][] result, int minLength, int nChannels,boolean printout, boolean full ,boolean store)
	{
		int[] start = new int[nChannels];
		int[] end 	= new int[nChannels];
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

			idx = 1;
			int counter = 0;

			while (idx < result.length && iterate)
			{
				if(printout)
					System.out.println("Idx: " + tempVector[idx]  + " counter: " + counter);
				if (tempVector[idx] <= 0)
				{
					if (counter > minLength)
					{
						end[channelIdx-1] = idx - 1;
						iterate = false;
						if(printout)
							System.out.println("End: " + end[channelIdx-1]);
					}
					counter = 0; // reset
				}else
				{
					if (tempVector[idx] < tempVector[idx - 1]) 
						counter++;
					if (tempVector[idx] > tempVector[idx - 1])
					{
						if (counter > minLength)
						{
							end[channelIdx-1] = idx - 1;
							iterate = false;
							if(printout)
								System.out.println("End: " + end[channelIdx-1]);
						}
						else
							counter = 0;
					}
				}

				if (counter == minLength)
				{
					start[channelIdx-1] = idx - minLength + 1;
					if(printout)
						System.out.println("Start: " + start[channelIdx-1]);
				}

				idx++;
				if (idx == result.length)
				{
					if (counter > minLength)
					{
						end[channelIdx-1] = idx-1;
						if(printout)
							System.out.println("End: " + end[channelIdx-1]);
					}
					else
						if(printout)
							System.out.println("counter: " + counter);
				}
			}
			if (start[channelIdx-1] == 1)
			{
				if (tempVector[0] > tempVector[1])
					start[channelIdx-1] = 0;
			}
			iterate = true;
			//	int currStart = start[channelIdx-1];
			double value = tempVector[start[channelIdx-1]];
			idx = start[channelIdx-1] - 1;
			while (iterate)
			{
				if (tempVector[idx] > 0 &&tempVector[idx] <= value)
				{
					start[channelIdx-1]++;					// step forward.
					if (start[channelIdx-1] < tempVector.length-1)
					{
						value = tempVector[start[channelIdx-1]]; // update value.
						idx = start[channelIdx-1] - 1;
					}else
						iterate = false;

				}
				idx--;
				if (idx == 0)
					iterate = false;

			}
			value = tempVector[end[channelIdx-1]];
			iterate = true;
			idx = end[channelIdx-1] + 1;
			while (iterate)
			{

				if (tempVector[idx] > 0 && tempVector[idx] >= value)
				{
					end[channelIdx-1]--;					// step forward.
					if (end[channelIdx-1] > 0)
					{
						value = tempVector[end[channelIdx-1]]; // update value.
						idx = end[channelIdx-1] + 1;
					}else
						iterate = false;

				}
				idx++;
				if (idx == tempVector.length)
					iterate = false;

			}
			if(printout)
			{
				System.out.println("start: " + start[channelIdx-1] + " end" + end[channelIdx-1]);
			}

			channelIdx++;

		}
		if(printout)
		{				
			channelIdx--;
			System.out.println("start: " + start[channelIdx-1] + " end: " + end[channelIdx-1] + " length: " + (end[channelIdx-1] - start[channelIdx-1] + 1));
		}
		for (int i = 0; i < nChannels; i++)
		{
			if (maxCounter < end[i] - start[i] + 1)
			{
				maxCounter = end[i] - start[i] + 1;
			}
		}
		if (full)
		{
			end[0] = result.length-1;
			start[0] = 0;
			maxCounter = end[0] - start[0] + 1;	
		}
		if (store)
		{
			for (int Ch = 1; Ch <= nChannels; Ch++)
			{				
				ij.Prefs.set("SMLocalizer.calibration.Biplane.center.Ch"+Ch,Math.round(result.length/2)-start[Ch-1]);			
			} 			
			ij.Prefs.savePreferences(); // store settings.
		}


		if (maxCounter < 0)
			maxCounter = 0;

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