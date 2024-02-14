package dasom;

import java.io.*;

class IOFiles {

    static boolean blank(int ascii) {
        switch (ascii) {
            case (int) ' ':
            case (int) '\t':
            case (int) '\r':
            case (int) '\n':
                return true;
            default:
                return false;
        }
    }

    static int fileNormalForm(String source, String destination) {
        int count;
        boolean isNum;
        String num;
        int ascii;

        count = 0;
        try {
            FileInputStream fis = new FileInputStream(source);
            FileOutputStream fos = new FileOutputStream(destination);
            BufferedOutputStream bos = new BufferedOutputStream(fos);
            DataOutputStream dos = new DataOutputStream(bos);

            isNum = false;
            num = "";
            while ((ascii = fis.read()) != -1) {
                if (blank(ascii) && !isNum); else if (!blank(ascii) && !isNum) {
                    isNum = true;
                    num += String.valueOf((char) ascii);
                } else if (!blank(ascii) && isNum) {
                    num += String.valueOf((char) ascii);
                } else if (blank(ascii) && isNum) {
                    count++;
                    dos.writeDouble(Double.parseDouble(num));
                    num = "";
                    isNum = false;
                }
            }
            if (isNum) {
                count++;
                dos.writeDouble(Double.parseDouble(num));
            }
            fis.close();
            dos.close();
        } catch (IOException e) {
            System.out.println("ERROR: " + e.toString());
        }
        return count;
    }

    static void arrayToFile(double array[][][], String file, boolean append) {
        try {
            FileOutputStream fos = new FileOutputStream(file, append);
            BufferedOutputStream bos = new BufferedOutputStream(fos);
            DataOutputStream dos = new DataOutputStream(bos);

            for (int i = 0; i < array.length; i++) {
                for (int j = 0; j < array[0].length; j++)
                    for (int k = 0; k<array[0][0].length; k++) {
                        dos.writeDouble(array[i][j][k]);
                }
            }
            dos.close();
        } catch (IOException e) {
            System.out.println("ERROR: " + e.toString());
        }
    }
    
    static void arrayToFile(double array[][], String file, boolean append) {
        try {
            FileOutputStream fos = new FileOutputStream(file, append);
            BufferedOutputStream bos = new BufferedOutputStream(fos);
            DataOutputStream dos = new DataOutputStream(bos);

            for (int i = 0; i < array.length; i++) {
                for (int j = 0; j < array[0].length; j++) {
                    dos.writeDouble(array[i][j]);
                }
            }
            dos.close();
        } catch (IOException e) {
            System.out.println("ERROR: " + e.toString());
        }
    }

    static void arrayToFile(double array[], String file, boolean append) {
        try {
            FileOutputStream fos = new FileOutputStream(file, append);
            BufferedOutputStream bos = new BufferedOutputStream(fos);
            DataOutputStream dos = new DataOutputStream(bos);

            for (int i = 0; i < array.length; i++) {
                dos.writeDouble(array[i]);
            }
            dos.close();
        } catch (IOException e) {
            System.out.println("ERROR: " + e.toString());
        }
    }

    static void arrayToFile(int array[][], String file, boolean append) {
        try {
            FileOutputStream fos = new FileOutputStream(file, append);
            BufferedOutputStream bos = new BufferedOutputStream(fos);
            DataOutputStream dos = new DataOutputStream(bos);

            for (int i = 0; i < array.length; i++) {
                for (int j = 0; j < array[0].length; j++) {
                    dos.writeInt(array[i][j]);
                }
            }
            dos.close();
        } catch (IOException e) {
            System.out.println("ERROR: " + e.toString());
        }
    }

    static void arrayToFile(int array[], String file, boolean append) {
        try {
            FileOutputStream fos = new FileOutputStream(file, append);
            BufferedOutputStream bos = new BufferedOutputStream(fos);
            DataOutputStream dos = new DataOutputStream(bos);

            for (int i = 0; i < array.length; i++) {
                dos.writeInt(array[i]);
            }
            dos.close();
        } catch (IOException e) {
            System.out.println("ERROR: " + e.toString());
        }
    }

    static double[] fileToArray(String file, int elements) {
        double value[] = new double[elements];
        try {
            FileInputStream fis = new FileInputStream(file);
            BufferedInputStream bis = new BufferedInputStream(fis);
            DataInputStream dis = new DataInputStream(bis);

            for (int i = 0; i < value.length; i++) {
                value[i] = dis.readDouble();
            }
            dis.close();
        } catch (IOException e) {
            System.out.println("ERROR: " + e.toString());
        }
        return value;
    }

    static double[][] fileToArray(String file, int rows, int columns) {
        double value[][] = new double[rows][columns];
        try {
            FileInputStream fis = new FileInputStream(file);
            BufferedInputStream bis = new BufferedInputStream(fis);
            DataInputStream dis = new DataInputStream(bis);

            for (int i = 0; i < value.length; i++) {
                for (int j = 0; j < value[0].length; j++) {
                    value[i][j] = dis.readDouble();
                }
            }
            dis.close();
        } catch (IOException e) {
            System.out.println("ERROR: " + e.toString());
        }
        return value;
    }

    static double[] fileStreamToArray(DataInputStream dis, int elements) {
        double value[] = new double[elements];
        try {
            for (int i = 0; i < value.length; i++) {
                value[i] = dis.readDouble();
            }
        } catch (IOException e) {
            System.out.println("ERROR: " + e.toString());
        }
        return value;
    }

    static double[][] fileStreamToArray(DataInputStream dis, int rows, int columns) {
        double value[][] = new double[rows][columns];
        try {
            for (int i = 0; i < value.length; i++) {
                for (int j = 0; j < value[0].length; j++) {
                    value[i][j] = dis.readDouble();
                }
            }
        } catch (IOException e) {
            System.out.println("ERROR: " + e.toString());
        }
        return value;
    }
    
    static double[][][] fileStreamToArray(DataInputStream dis, int x, int y, int z) {
        double value[][][] = new double[x][y][z];
        try {
            for (int i = 0; i < value.length; i++) {
                for (int j = 0; j < value[0].length; j++) {
                        for (int k = 0; k < value[0][0].length; k++) {
                            value[i][j][k] = dis.readDouble();
                        }
                }
            }
        } catch (IOException e) {
            System.out.println("ERROR: " + e.toString());
        }
        return value;
    }

    static int[] fileToIntArray(String file, int elements) {
        int value[] = new int[elements];
        try {
            FileInputStream fis = new FileInputStream(file);
            BufferedInputStream bis = new BufferedInputStream(fis);
            DataInputStream dis = new DataInputStream(bis);

            for (int i = 0; i < value.length; i++) {
                value[i] = dis.readInt();
            }
            dis.close();
        } catch (IOException e) {
            System.out.println("ERROR: " + e.toString());
        }
        return value;
    }

    static int[][] fileToIntArray(String file, int rows, int columns) {
        int value[][] = new int[rows][columns];
        try {
            FileInputStream fis = new FileInputStream(file);
            BufferedInputStream bis = new BufferedInputStream(fis);
            DataInputStream dis = new DataInputStream(bis);

            for (int i = 0; i < value.length; i++) {
                for (int j = 0; j < value[0].length; j++) {
                    value[i][j] = dis.readInt();
                }
            }
            dis.close();
        } catch (IOException e) {
            System.out.println("ERROR: " + e.toString());
        }
        return value;
    }

    static int[] fileStreamToIntArray(DataInputStream dis, int elements) {
        int value[] = new int[elements];
        try {
            for (int i = 0; i < value.length; i++) {
                value[i] = dis.readInt();
            }
        } catch (IOException e) {
            System.out.println("ERROR: " + e.toString());
        }
        return value;
    }

    static int[][] fileStreamToIntArray(DataInputStream dis, int rows, int columns) {
        int value[][] = new int[rows][columns];
        try {
            for (int i = 0; i < value.length; i++) {
                for (int j = 0; j < value[0].length; j++) {
                    value[i][j] = dis.readInt();
                }
            }
        } catch (IOException e) {
            System.out.println("ERROR: " + e.toString());
        }
        return value;
    }

    static void show(double pin[]) {
        for (int i = 0; i < pin.length; i++) {
            System.out.print(pin[i] + " ");
        }
        System.out.println();
    }

    static void showVert(double pin[]) {
        for (int i = 0; i < pin.length; i++) {
            System.out.println(pin[i]);
        }
    }

    static void show(double pin[][]) {
        for (int i = 0; i < pin.length; i++) {
            for (int j = 0; j < pin[0].length; j++) {
                System.out.print(pin[i][j] + " ");
            }
            System.out.println();
        }
    }
        
}
