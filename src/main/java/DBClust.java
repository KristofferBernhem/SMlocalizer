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
import java.util.List;

import org.apache.commons.math3.ml.clustering.Cluster;
import org.apache.commons.math3.ml.clustering.DBSCANClusterer;
import org.apache.commons.math3.ml.clustering.DoublePoint;

import ij.ImagePlus;
import ij.plugin.filter.Analyzer;
import ij.process.ByteProcessor;


/*
 * Change to update table with cluster information.
 * Create version for 3D data.
 */
public class DBClust {
	public static List<Cluster<DoublePoint>> Ident(double eps, int minPts, int pixelSize){
		ArrayList<Particle> InpParticle = TableIO.Load(); // Get current table data.
		List<DoublePoint> points = new ArrayList<DoublePoint>();

		for (int i = 0; i < InpParticle.size(); i++){
			double[] p = new double[2];
			if (InpParticle.get(i).include == 1){
				p[0] = InpParticle.get(i).x;
				p[1] = InpParticle.get(i).y;
				points.add(new DoublePoint(p));
			}
		}
		DBSCANClusterer<DoublePoint> DB = new DBSCANClusterer<DoublePoint>(eps, minPts);
		List<Cluster<DoublePoint>> cluster = DB.cluster(points);
		int ClustIdx 			= 1;
		int[] IndexList 		= new int[InpParticle.size()];
		int Width 				= 0;
		int Height 				= 0;
		int Pad					= 100;
		for(Cluster<DoublePoint> c: cluster){
			for (int j = 0; j < c.getPoints().size(); j++){
				DoublePoint p = c.getPoints().get(j);
				double[] Coord = p.getPoint();
				for (int i = 0; i < InpParticle.size();i++){
					Particle tempParticle = InpParticle.get(i);
					if(tempParticle.x == Coord[0] && tempParticle.y == Coord[1]){
						IndexList[i] = ClustIdx; 
					}
					if (Math.round(tempParticle.x) > Width){
						Width = (int) Math.round(tempParticle.x) + Pad;
					}
					if (Math.round(tempParticle.y) > Height){
						Height = (int) Math.round(tempParticle.y) + Pad;
					}

				}				
			}
			ClustIdx++;
		}  
		Width = Math.round(Width/pixelSize);
		Height = Math.round(Height/pixelSize);
		if(Width > 0 && Height > 0){

			ByteProcessor IP  = new ByteProcessor(Width,Height);			
			for (int x = 0; x < Width; x++){
				for (int y = 0; y < Height; y++){
					IP.putPixel(x, y, 0); // Set all data points to 0 as start.
				}

			}
			ij.measure.ResultsTable tab = Analyzer.getResultsTable();
			tab.reset();		
			for (int i = 0; i < InpParticle.size(); i++){
				tab.incrementCounter();
				tab.addValue("Cluster", IndexList[i]);
				tab.addValue("x0", InpParticle.get(i).x);
				tab.addValue("y0", InpParticle.get(i).y);
				tab.addValue("z0", InpParticle.get(i).z);
				tab.addValue("frame", InpParticle.get(i).frame);
				tab.addValue("channel", InpParticle.get(i).channel);
				tab.addValue("sigma_x", InpParticle.get(i).sigma_x);
				tab.addValue("sigma_y", InpParticle.get(i).sigma_y);
				tab.addValue("sigma_z", InpParticle.get(i).sigma_z);
				tab.addValue("precision_x", InpParticle.get(i).precision_x);
				tab.addValue("precision_y", InpParticle.get(i).precision_y);
				tab.addValue("precision_z", InpParticle.get(i).precision_z);
				tab.addValue("r_square", InpParticle.get(i).r_square);
				tab.addValue("photons", InpParticle.get(i).photons);
				if (IndexList[i] == 0){
					tab.addValue("include", 0);
				}else{
					tab.addValue("include", 1);
					int x = (int) Math.round(InpParticle.get(i).x/pixelSize);
					int y = (int) Math.round(InpParticle.get(i).y/pixelSize);		
					IP.putPixel(x, y, 1);
				}
			}
			tab.show("Results");
			ImagePlus Image = new ImagePlus("ClusterData",IP);
			Image.setImage(Image);
			Image.show(); 														// Make visible
		}
		return cluster; 
	}
	public static int getIdx(double x, double y, int width, int height){
		int Idx = (int) ( ((y+1)*height) - (width-(x+1)));		
		return Idx;
	}

}
