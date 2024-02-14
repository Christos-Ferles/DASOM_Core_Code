package dasom;

public class FunctionValueSequence {
    
    /*
    Construction of a sequence of values originating from a linear function.
    The steps variable must be different from 1 and should attain positive
    values, steps>1. (Refer also to LR & NB.xls)
    */
    public static double[] linear(int steps, double initialValue, double finalValue) {
        double sequenceOfValues[] = new double[steps];
        
        for (int i = 0; i<steps; i++)
            sequenceOfValues[i] = (finalValue-initialValue)*i/(double)(steps-1)+initialValue;
        
        return sequenceOfValues;
    }
    
    /*
    Construction of a sequence of values originating from an exponential function.
    The steps variable must be different from 1 and should attain positive
    values, steps>1. Moreover the initialValue variable must be different from
    zero and both intialValue and finalValue must have the same sign. After
    referring to LR & NB.xls it can be noticed that the point distribution is not
    fully balanced but it is better in comparison to the inverse case by far.
    */
    public static double[] exponential(int steps, double initialValue, double finalValue) {
        double sequenceOfValues[] = new double[steps];
        double factor = initialValue;
        double coefficient = Math.log(finalValue/initialValue)/(double)(steps-1);
        
        for (int i = 0; i<steps; i++)
            sequenceOfValues[i] = factor*Math.exp(coefficient*i);
        
        return sequenceOfValues;
    }
    
    /*
    Construction of a sequence of values originating from an inverse function:
              numerator       
    y = ----------------------
         partialDominator + x 
    Obviously, the steps variable must have positive values, steps>0. After
    referring to LR & NB.xls it is evident that the distribution of points is
    not balanced, since their majority is concentrated towards lower y values.
    Possibly this is a reason why it should be avoided during initial rough
    adaptations.
    */
    public static double[] inverse(int steps, double initialValue, double finalValue) {
        double sequenceOfValues[] = new double[steps];      
        double numerator = (steps-1)*finalValue*initialValue/(initialValue-finalValue);
        double partialDenominator = (steps-1)*finalValue/(initialValue-finalValue);
          
        for (int i = 0; i<steps; i++)
            sequenceOfValues[i] = numerator/(partialDenominator+i);
        
        return sequenceOfValues;
    }
    
}
