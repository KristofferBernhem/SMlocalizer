
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

/*
 * Hold all information for fitting a single event.
 */

	


public class fitParameters {
	int[] Center;
	int[] data;
	int channel;
	int frame;
	int pixelsize;
	int windowWidth;
	int totalGain;
	public fitParameters()
	{
		
	}
	public fitParameters(int[] Center, int[] data, int channel, int frame, int pixelsize, int windowWidth, int totalGain){
		this.Center 	= Center; 			// X and Y center coordinates.
		this.data 		= data; 			// array of pixelvalues.
		this.channel 	= channel;  		// channel number.
		this.frame 		= frame;			// frame number.
		this.pixelsize 	= pixelsize;		// pixelsize.
		this.windowWidth = windowWidth; 	// width of extracted region.
		this.totalGain	= totalGain; 		// total amplification from photon to pixel value in input image. Ask camera manufacturer for how to obtain this.
	}
}
