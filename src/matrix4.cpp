//
//  matrix4.cpp
//  Assignment3
//
//  Created by memo2 on 10/27/23.
//

#include <stdio.h>
#include "matrix4.h"

matrix matrix::translate( float tx, float ty,  float tz){
    vec u = vec(1,0,0,tx);
    vec v = vec(0,1,0,ty);
    vec w = vec(0,0,1,tz);
    vec t = vec(0,0,0,1);
    matrix m = matrix(u,v,w,t);
    return m;
}

matrix matrix::scale(float sx, float sy, float sz){
    vec u = vec(sx,0,0,0);
    vec v = vec(0,sy,0,0);
    vec w = vec(0,0,sz,0);
    vec t = vec(0,0,0,1);
    matrix m = matrix(u,v,w,t);
    return m;
}
matrix matrix::RotateZ( double theta){
    vec u = vec(cos(theta),-sin(theta),0,0);
    vec v = vec(sin(theta),cos(theta),0,0);
    vec w = vec(0,0,1,0);
    vec t = vec(0,0,0,1);
    matrix m = matrix(u,v,w,t);
    return m;
}

matrix matrix::RotateX( double theta){
    vec u = vec(1,0,0,0);
    vec v = vec(0,cos(theta),-sin(theta),0);
    vec w = vec(0,sin(theta),cos(theta),0);
    vec t = vec(0,0,0,1);
    matrix m = matrix(u,v,w,t);
    return m;
}

matrix matrix::RotateY( double theta){
    vec u = vec(cos(theta),0,sin(theta),0);
    vec v = vec(0,1,0,0);
    vec w = vec(-sin(theta),0,cos(theta),0);
    vec t = vec(0,0,0,1);
    matrix m = matrix(u,v,w,t);
    return m;
}

matrix matrix::SetupCamTransformation(vec cam, vec uVec, vec vVec,vec wVec){
    // make sure these vectors are normalized
    vec fourth = vec(-(uVec*cam), -(vVec*cam),-(wVec*cam),1);
    vec f = vec(0,0,0,1);
    matrix viewMatrix = matrix(uVec,vVec,wVec,f);
    viewMatrix.m[0][3] = fourth.x();
    viewMatrix.m[1][3] = fourth.y();
    viewMatrix.m[2][3] = fourth.z();
    return viewMatrix;
}

matrix matrix::SetupViewportTransformation(float nx, float ny){ //  M_vp
    matrix m;
    m.m[0][0] = nx/2; m.m[1][1] = ny/2;
    m.m[0][3] = (nx-1)/2; m.m[1][3] = (ny-1)/2;
    return m;
}

matrix matrix::SetupOrthProjTransformation(float l, float r, float b, float t, float f, float n){
    matrix m;
    m.m[0][0] = 2/(r-l); m.m[1][1] = 2/(t-b); m.m[2][2] = 2/(n-f);
    m.m[0][3] = -((r+l)/(r-l)); m.m[1][3] = -((t+b)/(t-b)); m.m[2][3] = -((n+f)/(n-f));
    return m;
}
matrix matrix::SetupPersProjTransformation(float l, float r, float b, float t, float f, float n){
    matrix p;
    p.m[0][0] = p.m[1][1] = n; p.m[3][3] = 0;
    p.m[2][2] = n+f; p.m[2][3] = -f*n; p.m[3][2]= 1;
    matrix M_orth = SetupOrthProjTransformation(l, r, b, t, f, n);
    matrix M_per = M_orth*p;
    return p;
}

matrix matrix::SetupModelMatrix(vec translate, double rotationAngle, vec rotationAxis, vec scale){
    matrix m;
    matrix translation = m.translate(translate.x(), translate.y(), translate.z());
    
    matrix rotation;
    float cosA = cos(rotationAngle);
    float sinA = sin(rotationAngle);
    float oneMinusCosA = 1.0f - cosA;
    rotation.m[0][0] = cosA + rotationAxis.v[0] * rotationAxis.v[0] * oneMinusCosA;
    rotation.m[1][0] = rotationAxis.x() * rotationAxis.y() * oneMinusCosA - rotationAxis.z() * sinA;
    rotation.m[2][0] = rotationAxis.x() * rotationAxis.z() * oneMinusCosA + rotationAxis.y() * sinA;

    rotation.m[0][1] = rotationAxis.y() * rotationAxis.x() * oneMinusCosA + rotationAxis.z() * sinA;
    rotation.m[1][1] = cosA + rotationAxis.y() * rotationAxis.y() * oneMinusCosA;
    rotation.m[2][1] = rotationAxis.y() * rotationAxis.z() * oneMinusCosA - rotationAxis.x() * sinA;

    rotation.m[0][2] = rotationAxis.z() * rotationAxis.x() * oneMinusCosA - rotationAxis.y() * sinA;
    rotation.m[1][2] = rotationAxis.z() * rotationAxis.y() * oneMinusCosA + rotationAxis.x() * sinA;
    rotation.m[2][2] = cosA + rotationAxis.z() * rotationAxis.z() * oneMinusCosA;
    
    matrix scaling = m.scale(scale.x(), scale.y(), scale.z());
    
    return scaling * (rotation * translation);   // not validated
}

matrix matrix::arcBall(float first_x, float first_y, float current_x, float current_y){
    // calculate z0, z1
    float z0 = 0.0f; float z1 = 0.0f;
    
    if ((pow(first_x,2)+pow(first_y,2)<= 1) && (pow(current_x,2)+pow(current_y,2)<= 1) ){
        z0 = sqrt(1-pow(first_x,2)-pow(first_y,2));
        z1 = sqrt(1-pow(current_x,2)-pow(current_y,2));
    }
//    }
//    if ((pow(current_x,2)+pow(current_y,2)<= 1)){
//        z1 = sqrt(1-pow(current_x,2)-pow(current_y,2));
//    }
    
    // u = a x b .. what if a and b  parallel?
    vec a = vec(first_x, first_y, z0, 0);
    vec b = vec(current_x, current_y, z1, 0);
    a.print(); b.print();
    vec u = (a.cross(b)).normalize(); // not sure
    u.print();
    
    // theta  = arcos(a.b/|a||b|)
    float f  = (a*b)/(a.length()*b.length());
    double th;
    th = acos(std::min(1.0f,f));
    std::cout << th << std::endl;
    
    // create the rotation matrix
    
    matrix m;
    
    double ux, uy, uz;
    ux = u.x();uy = u.y(); uz = u.z();
    
    // first row
    m.m[0][0] =pow(ux,2)+ (1-pow(ux,2))*cos(th);
    m.m[0][1] =ux*uy*(1-cos(th))-uz*sin(th);
    m.m[0][2] =ux*uz*(1-cos(th))+uy*sin(th);
    // second row
    m.m[1][0] =uy*ux*(1-cos(th))+uz*sin(th);
    m.m[1][1] =pow(uy,2)+ (1-pow(uy,2))*cos(th);
    m.m[1][2] =uy*uz*(1-cos(th))-ux*sin(th);
    // third row
    m.m[2][0] =uz*ux*(1-cos(th))-uy*sin(th);
    m.m[2][1] =uz*uy*(1-cos(th))+ux*sin(th);
    m.m[2][2] =pow(uz,2)+ (1-pow(uz,2))*cos(th);
    //m.transpose();
    //matrix rotatex = m.RotateX(M_PI/4);
    
    return m;
}

