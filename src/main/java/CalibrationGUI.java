import java.util.ArrayList;

import ij.ImagePlus;
import ij.WindowManager;
import ij.process.ImageStatistics;

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
 * @author kristoffer.bernhem
 */
@SuppressWarnings("serial")
public class CalibrationGUI extends javax.swing.JFrame {

    /**
     * Creates new form NewJFrame
     */
    public CalibrationGUI() {
        initComponents();
    }


    // <editor-fold defaultstate="collapsed" desc="Generated Code">                          
    private void initComponents() {

        jLabel1 = new javax.swing.JLabel();
        algorithmSelect = new javax.swing.JComboBox<>();
        jLabel2 = new javax.swing.JLabel();
        jLabel3 = new javax.swing.JLabel();
        pixelSize = new javax.swing.JTextField();
        stepSize = new javax.swing.JTextField();
        Process = new javax.swing.JButton();

      //  setDefaultCloseOperation(javax.swing.WindowConstants.EXIT_ON_CLOSE);

        jLabel1.setFont(new java.awt.Font("Times New Roman", 0, 24)); // NOI18N
        jLabel1.setText("SMLocalizer 3D calibration");

        algorithmSelect.setFont(new java.awt.Font("Times New Roman", 3, 12)); // NOI18N
        algorithmSelect.setModel(new javax.swing.DefaultComboBoxModel<>(new String[] { "PRILM", "Double Helix", "Biplane" , "Astigmatism"}));

        jLabel2.setFont(new java.awt.Font("Times New Roman", 0, 11)); // NOI18N
        jLabel2.setText("Pixelsize [nm]:");

        jLabel3.setFont(new java.awt.Font("Times New Roman", 0, 11)); // NOI18N
        jLabel3.setText("Step size Z [nm]:");

        pixelSize.setFont(new java.awt.Font("Times New Roman", 0, 11)); // NOI18N
        pixelSize.setHorizontalAlignment(javax.swing.JTextField.CENTER);
        pixelSize.setText("100");
        pixelSize.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                pixelSizeActionPerformed(evt);
            }
        });

        stepSize.setFont(new java.awt.Font("Times New Roman", 0, 11)); // NOI18N
        stepSize.setHorizontalAlignment(javax.swing.JTextField.CENTER);
        stepSize.setText("10");

        Process.setFont(new java.awt.Font("Times New Roman", 3, 12)); // NOI18N
        Process.setText("Process");
        Process.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                ProcessActionPerformed(evt);
            }
        });

        javax.swing.GroupLayout layout = new javax.swing.GroupLayout(getContentPane());
        getContentPane().setLayout(layout);
        layout.setHorizontalGroup(
            layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(layout.createSequentialGroup()
                .addContainerGap()
                .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                    .addComponent(algorithmSelect, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                    .addGroup(layout.createSequentialGroup()
                        .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                            .addComponent(jLabel3)
                            .addComponent(jLabel2))
                        .addGap(18, 18, 18)
                        .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.TRAILING)
                            .addComponent(pixelSize, javax.swing.GroupLayout.Alignment.LEADING, javax.swing.GroupLayout.PREFERRED_SIZE, 40, javax.swing.GroupLayout.PREFERRED_SIZE)
                            .addComponent(stepSize, javax.swing.GroupLayout.Alignment.LEADING, javax.swing.GroupLayout.PREFERRED_SIZE, 40, javax.swing.GroupLayout.PREFERRED_SIZE)))
                    .addComponent(Process)
                    .addComponent(jLabel1))
                .addContainerGap(javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE))
        );
        layout.setVerticalGroup(
            layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(layout.createSequentialGroup()
                .addContainerGap()
                .addComponent(jLabel1)
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addComponent(algorithmSelect, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.BASELINE)
                    .addComponent(jLabel2)
                    .addComponent(pixelSize, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE))
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.BASELINE)
                    .addComponent(jLabel3)
                    .addComponent(stepSize, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE))
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addComponent(Process)
                .addContainerGap(javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE))
        );

        pack();
    }// </editor-fold>                        

    private void pixelSizeActionPerformed(java.awt.event.ActionEvent evt) {                                          
    }                                         

    private void ProcessActionPerformed(java.awt.event.ActionEvent evt) {                               
    	/*
    	 * Localize all events in all frames.
    	 */
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
		ImageStatistics IMstat 	= image.getStatistics();
		int[] MinLevel 			= {(int) (IMstat.max*0.6)};
		int[] gWindow 			= {5};
		int[] inputPixelSize 	= {Integer.parseInt(pixelSize.getText())};
		int[] minPosPixels 		= {12};
		int[] totalGain 		= {100};
		int selectedModel 		= 0; // CPU
    	localizeAndFit.run(MinLevel, gWindow, inputPixelSize, minPosPixels, totalGain, selectedModel);
    	/*
    	 * clean out fits based on goodness of fit:
    	 */
    	boolean[][] include = new boolean[7][1];
    	include[0][0] 		= false;
    	include[1][0] 		= false;
    	include[2][0] 		= false;
    	include[3][0] 		= true;
    	include[4][0] 		= false;
    	include[5][0] 		= false;
    	include[6][0] 		= false;    	
		double[][] lb 		= new double[7][1];
		lb[0][0]			= 0;
		lb[1][0]			= 0;
		lb[2][0]			= 0;
		lb[3][0]			= 0.7;
		lb[4][0]			= 0;
		lb[5][0]			= 0;
		lb[6][0]			= 0;
		double[][] ub 		= new double[7][1];
		ub[0][0]			= 0;
		ub[1][0]			= 0;
		ub[2][0]			= 0;
		ub[3][0]			= 1.0;
		ub[4][0]			= 0;
		ub[5][0]			= 0;
		ub[6][0]			= 0;
		cleanParticleList.run(lb,ub,include);
		cleanParticleList.delete();
    	
    	/*
    	 * Set z values based on frame and z-stepsize.
    	 */
		boolean[] renderCh = {true};
		int[] DesiredPixelSize = {5,10};
		boolean gSmoothing = true;
		ArrayList<Particle> result = TableIO.Load();
		int zStep = Integer.parseInt(stepSize.getText());
		for (int i = 0; i < result.size(); i++)
		{
			result.get(i).z = (result.get(i).frame-1)*zStep;
		}
		TableIO.Store(result);
		//RenderIm.run(renderCh, DesiredPixelSize, gSmoothing);
    	
    	/*
    	 * translate into calibration data based on method:
    	 */
		result = TableIO.Load();
		int minSqdist = 600*600;
    	if(algorithmSelect.getSelectedIndex() == 0)
    	{ // PRILM
    		System.out.println("PRILM");
    		int id = 2;
    		int counter = 0;
    		for (int i = 0; i < result.size(); i++)
    		{
    			counter = 0;
    			if (result.get(i).include == 1) // if the current entry is within ok range and has not yet been assigned.
    			{
    				int idx = i + 1;
    				while (idx < result.size() && result.get(i).channel == result.get(idx).channel)
    				{
    					if (result.get(i).channel == result.get(idx).channel &&
    							result.get(i).z == result.get(idx).z)
    					{
    						if (((result.get(i).x - result.get(idx).x)*(result.get(i).x - result.get(idx).x) +
    							(result.get(i).y - result.get(idx).y)*(result.get(i).y - result.get(idx).y)) < minSqdist)
    						{
    							result.get(idx).include = id;
    							result.get(i).include = id;
    							counter ++;
    						}
    					}
    					idx ++;
    				}    				    				
    				if (counter > 1)
    				{
    					for (int j = 0; j < result.size(); j++)
    					{
    						if (result.get(j).include == id)
    						{
    							result.get(j).include = 0;
    						}
    					}
    				}
    				id++;
    			}
    		}
    		TableIO.Store(result);
      	}else 
      	if(algorithmSelect.getSelectedIndex() == 1)	
      	{ // Double helix
    		System.out.println("Double helix");      		
      	}else
      	if(algorithmSelect.getSelectedIndex() == 2)
      	{ // Biplane
    		System.out.println("Biplane");
      	}else
      	if(algorithmSelect.getSelectedIndex() == 3)
      	{ // Astigmatism
    		System.out.println("Astigmatism");
      	}
    }                                       

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
            java.util.logging.Logger.getLogger(CalibrationGUI.class.getName()).log(java.util.logging.Level.SEVERE, null, ex);
        } catch (InstantiationException ex) {
            java.util.logging.Logger.getLogger(CalibrationGUI.class.getName()).log(java.util.logging.Level.SEVERE, null, ex);
        } catch (IllegalAccessException ex) {
            java.util.logging.Logger.getLogger(CalibrationGUI.class.getName()).log(java.util.logging.Level.SEVERE, null, ex);
        } catch (javax.swing.UnsupportedLookAndFeelException ex) {
            java.util.logging.Logger.getLogger(CalibrationGUI.class.getName()).log(java.util.logging.Level.SEVERE, null, ex);
        }
        //</editor-fold>
        //</editor-fold>
        //</editor-fold>
        //</editor-fold>

        /* Create and display the form */
        java.awt.EventQueue.invokeLater(new Runnable() {
            public void run() {
                new CalibrationGUI().setVisible(true);
            }
        });
    }

    // Variables declaration - do not modify                     
    private javax.swing.JButton Process;
    private javax.swing.JComboBox<String> algorithmSelect;
    private javax.swing.JLabel jLabel1;
    private javax.swing.JLabel jLabel2;
    private javax.swing.JLabel jLabel3;
    private javax.swing.JTextField pixelSize;
    private javax.swing.JTextField stepSize;
    // End of variables declaration                   
}
