Quick guide to getting EM-CaMML started (updated 20-01-2020):

EM-CaMML is based on the original version of CaMML: https://github.com/rodneyodonnell/CaMML
It is still under development and we suggest using Eclipse to open EM-CaMML: </br>

1. Make to clone to Eclipse from GitHub.</br>
2. In the "Project Properties" dialog, choose the "Java Build Path" link, then click on the "Libraries" tab, </br>
   and then click on the "Add External Jars" link. Navigate to the NeticaJ directory (e.g., C:\NeticaJ_418\bin\x64_bin) </br>
   and select NeticaJ.jar.
3. In the "Run As" dialog, go to the "Arguments" tab and in the "VM Arguments" window create the folling argument:
   -Djava.library.path=your NeticaJ path (e.g., -Djava.library.path=C:\NeticaJ_418\bin\x64_bin).
4. Windows only: Still in the "Run As" dialog, go to the "Environment" tab and create a new "PATH" variable with value:
   your NeticaJ path;%PATH% (e.g., C:\NeticaJ_418\bin\x64_bin;%PATH%)

Please note we suggest to use .arff format data as input. 

It will be very appreciated for any feedback and comments. Please report any bug to Xuhui Zhang (zxh298@gmail.com)


