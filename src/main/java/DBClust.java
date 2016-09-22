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

import ij.plugin.filter.Analyzer;



/*
 * TODO:
 * Change to update table with cluster information.
 * Create version for 3D data.
 * Create channel loop.
 */
public class DBClust {
	public static void Ident(double[] eps, int[] minPts, int[] pixelSize, boolean[] doCluster){
		ArrayList<Particle> InpParticle = TableIO.Load(); // Get current table data.
		int Width 				= 0;
		int Height 				= 0;
		ij.measure.ResultsTable tab = Analyzer.getResultsTable();
		tab.reset();		
		for (int Ch = 1; Ch <=InpParticle.get(InpParticle.size()-1).channel; Ch++){
			if (doCluster[Ch-1])
			{
				List<DoublePoint> points = new ArrayList<DoublePoint>();
		
				for (int i = 0; i < InpParticle.size(); i++){
					double[] p = new double[2];
					if (InpParticle.get(i).include == 1 && InpParticle.get(i).channel == Ch){
						p[0] = InpParticle.get(i).x;
						p[1] = InpParticle.get(i).y;
						points.add(new DoublePoint(p));
					}
				}
				DBSCANClusterer<DoublePoint> DB = new DBSCANClusterer<DoublePoint>(eps[Ch-1], minPts[Ch-1]);
				List<Cluster<DoublePoint>> cluster = DB.cluster(points);
				int ClustIdx 			= 1;
				int[] IndexList 		= new int[InpParticle.size()];
	
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
				Width = Math.round(Width/pixelSize[Ch-1]) + 1;
				Height = Math.round(Height/pixelSize[Ch-1] + 1);
				
				for (int i = 0; i < InpParticle.size(); i++){
					
					if(InpParticle.get(i).channel == Ch)
					{
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
					}
				}
			}
		} // Channel loop.
		tab.show("Results");
		
		RenderIm.run(pixelSize,false);
		
	}
	public static int getIdx(double x, double y, int width, int height){
		int Idx = (int) ( ((y+1)*height) - (width-(x+1)));		
		return Idx;
	}

}
