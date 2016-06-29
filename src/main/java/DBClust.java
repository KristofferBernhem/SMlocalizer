import java.util.ArrayList;
import java.util.List;

import org.apache.commons.math3.ml.clustering.Cluster;
import org.apache.commons.math3.ml.clustering.DBSCANClusterer;
import org.apache.commons.math3.ml.clustering.DoublePoint;

public class DBClust {
	
	public static List<Cluster<DoublePoint>> Ident(double eps, int minPts){
		ArrayList<Particle> InpParticle = TableIO.Load(); // Get current table data.
		List<DoublePoint> points = new ArrayList<DoublePoint>();

		for (int i = 0; i < InpParticle.size(); i++){
			double[] p = new double[2];
			p[0] = InpParticle.get(i).x;
			p[1] = InpParticle.get(i).y;
			points.add(new DoublePoint(p));
		}
		DBSCANClusterer<DoublePoint> DB = new DBSCANClusterer<DoublePoint>(eps, minPts);
		return DB.cluster(points); 
	}


}
