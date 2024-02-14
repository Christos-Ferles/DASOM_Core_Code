package dasom;

import java.util.*;

class DataManipulation {
    
    /*
    Input data are shuffled the desired number of times (swaps). Following the
    practice data are considered to consist of vectors stored in rows where each
    column represents a feature/element. A good estimate for swaps appears to be:
    #samples*1000    
    */
    static void shuffle(double data[][], int swaps) {
        Random index = new Random();
        int firstIndex, secondIndex;
        double temporary[];
        
        for (int i =0; i<swaps; i++) {
            firstIndex = index.nextInt(data.length);
            secondIndex = index.nextInt(data.length);
            temporary = data[firstIndex];
            data[firstIndex] = data[secondIndex];
            data[secondIndex] = temporary;
        }
    }
    
    /*
    Finds the per column minimum and maximum values of the input array. The
    returned 2x(#features) array contains in its first row the minimum values
    and in its second row the maximum values.
    */
    static double[][] perColumnMinMax(double data[][]) {
        int samples = data.length;
        int features = data[0].length;
        double limits[][] = new double[2][data[0].length];
        
        for (int j = 0; j<features; j++)
            limits[0][j] = limits[1][j] = data[0][j];
        for (int i = 1; i<samples; i++)
            for (int j = 0; j<features; j++) {
                if (limits[0][j]>data[i][j])
                    limits[0][j] = data[i][j];
                else if (limits[1][j]<data[i][j])
                    limits[1][j] = data[i][j];
            }                       
        return limits;
    }
    
    /*
    The values of the elements of each column are normalized to either the [0,1]
    range, if the zeroMinusOne is true, or the [-1,1] range, if the zeroMinusOne
    is false. This normalization is based on the provided minimum and maximum
    per column values which are supposed to be contained in the limits[][]
    array (this array contains in its first row the minimum value and in its
    second row the maximum values of each column).
    */
    static double[][] adjustPerColumnValueRange(double data[][], double limits[][],
                                                boolean zeroOrMinusOne) {
        int samples = data.length;
        int features = data[0].length;
        double normalizedData[][] = new double[samples][features];
        
        for (int i = 0; i<samples; i++)
            for (int j = 0; j<features; j++)
                normalizedData[i][j] = (data[i][j]-limits[0][j])/(limits[1][j]-limits[0][j]);
        if (!zeroOrMinusOne)
            for (int i = 0; i<samples; i++)
                for (int j = 0; j<features; j++)
                    normalizedData[i][j] = 2.0*normalizedData[i][j]-1.0;
        return normalizedData;    
    }
        
}
