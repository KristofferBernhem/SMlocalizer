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
import static jcuda.driver.JCudaDriver.cuMemAlloc;
import static jcuda.driver.JCudaDriver.cuMemcpyHtoD;

import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.CUdeviceptr;
/*
 * Contain help classes for jcuda implementation.
 */
public class CUDA {
	
	static CUdeviceptr copyToDevice(float hostData[])
	{
	    CUdeviceptr deviceData = new CUdeviceptr();
	    cuMemAlloc(deviceData, hostData.length * Sizeof.FLOAT);
	    cuMemcpyHtoD(deviceData, Pointer.to(hostData), hostData.length * Sizeof.FLOAT);
	    return deviceData;
	}
	
	static CUdeviceptr copyToDevice(int hostData[])
	{
	    CUdeviceptr deviceData = new CUdeviceptr();
	    cuMemAlloc(deviceData, hostData.length * Sizeof.INT);
	    cuMemcpyHtoD(deviceData, Pointer.to(hostData), hostData.length * Sizeof.INT);
	    return deviceData;
	}
	
	static CUdeviceptr allocateOnDevice(int hostDatasize)
	{
	    CUdeviceptr deviceData = new CUdeviceptr();
	    cuMemAlloc(deviceData, hostDatasize * Sizeof.INT);	  
	    return deviceData;
	}
	static CUdeviceptr allocateOnDevice(float hostDatasize)
	{
	    CUdeviceptr deviceData = new CUdeviceptr();
	    cuMemAlloc(deviceData, (int)hostDatasize * Sizeof.FLOAT);	  
	    return deviceData;
	}
	static CUdeviceptr allocateOnDevice(short hostDatasize)
	{
	    CUdeviceptr deviceData = new CUdeviceptr();
	    cuMemAlloc(deviceData, hostDatasize * Sizeof.SHORT);	  
	    return deviceData;
	}
}
