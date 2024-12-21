// Gmsh project created on Sun Nov  3 14:15:23 2024
SetFactory("OpenCASCADE");

lc = 0.05;

Point(1) = {2.2, 0, 0, lc};
Point(2) = {2.2, 0.41, 0, lc};
Point(3) = {0, 0.41, 0, lc};
Point(4) = {0, 0, 0, lc};

Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 1};

Curve Loop(1) = {1,2,3,4};

clc = lc / 2;
Point(5) = {0.2, 0.15, 0, clc};
Point(6) = {0.2, 0.2, 0, clc};
Point(7) = {0.25, 0.2, 0, clc};
Point(8) = {0.2, 0.25, 0, clc};
Point(9) = {0.15, 0.2, 0, clc};

Circle(5) = {5, 6, 7};
Circle(6) = {7, 6, 8};
Circle(7) = {8, 6, 9};
Circle(8) = {9, 6, 5};
Curve Loop(2) = {5,6,7,8};

Plane Surface(1) = {1,2};

Physical Curve("Right") = {1};
Physical Curve("Top") = {2};
Physical Curve("Left") = {3};
Physical Curve("Bottom") = {4};
Physical Curve("Circle") = {5,6,7,8};
Physical Surface("Interior") = {2,1};