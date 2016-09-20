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

public class cleanParticleList {

	public static void run(double[][] lb, double[][] ub, boolean[][] Include){
		// Pull in user data.
		/*	            Include[0][id] = doPhotonCountChList.getItem(id).isSelected();
            Include[1][id] = doSigmaXYChList.getItem(id).isSelected();
            Include[2][id] = doSigmaZChList.getItem(id).isSelected();
            Include[3][id] = doRsquareChList.getItem(id).isSelected();
            Include[4][id] = doPrecisionXYChList.getItem(id).isSelected();
            Include[5][id] = doPrecisionZChList.getItem(id).isSelected();

		 */
		ArrayList<Particle> Results = TableIO.Load();
		int Ch = 0;
		int remove = 0;
		for (int i = 0; i < Results.size(); i++){
			Ch = (int) Results.get(i).channel-1;
			remove = 0;
			if (Include[0][Ch]== true)
			{
				
				if(Results.get(i).photons >= lb[0][Ch] &&
						Results.get(i).photons <= ub[0][Ch]){

				}else
					remove++;				
			}
			
			if (Include[1][Ch]){
				
				if(Results.get(i).sigma_x >= lb[1][Ch] &&
						Results.get(i).sigma_x <= ub[1][Ch] &&
						Results.get(i).sigma_y >= lb[1][Ch] &&			
						Results.get(i).sigma_y <= ub[1][Ch]){

				}else
					remove++;
			}
			if (Include[2][Ch]){ 
				if(Results.get(i).sigma_z >= lb[2][Ch] &&
						Results.get(i).sigma_z <= ub[2][Ch]){

				}else
					remove++;
			}
			if (Include[3][Ch]){
				if(Results.get(i).r_square >= lb[3][Ch] &&			
						Results.get(i).r_square <= ub[3][Ch]){

				}else
					remove++;
			}
			if (Include[4][Ch]){
				if(Results.get(i).precision_x >= lb[4][Ch] &&
						Results.get(i).precision_x <= ub[4][Ch] && 
						Results.get(i).precision_y >= lb[4][Ch] &&
						Results.get(i).precision_y <= ub[4][Ch]){

				}else
					remove++;
			}
			if (Include[5][Ch]){ 
				if(Results.get(i).precision_z >= lb[5][Ch] &&
						Results.get(i).precision_z <= ub[5][Ch]){

				}else
					remove++;
			}			
			if(remove == 0){
				Results.get(i).include = 1; // Include particle.
			}else {
				Results.get(i).include = 0; // Exclude particle
			}			
		}
		TableIO.Store(Results);

	}
	public static void delete(){ // Remove entries outside of parameter range, first call the run function.
		ArrayList<Particle> Results = TableIO.Load();
		ArrayList<Particle> cleanedList = new ArrayList<Particle>();
		for (int i = 0; i < Results.size(); i++){
			if (Results.get(i).include == 1){
				cleanedList.add(Results.get(i));
			}
		}
		TableIO.Store(cleanedList);
	}
}
