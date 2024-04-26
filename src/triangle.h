//
//  triangle.h
//  Assignment3
//
//  Created by memo2 on 10/27/23.
//

#ifndef triangle_h
#define triangle_h

class triangle{
public:
    vec a;      // vertex 1
    vec b;      // vertex 2
    vec c;      // vertex 3
    vec color;
    vec normal; // normal of the face
    matrix transformation;
    
    triangle(){
        a = vec(0,0,0,1);
        b = vec(0,0,0,1);
        c = vec(0,0,0,1);
        color = vec(0.5,0.5,0.5,0);
        transformation = matrix();
    }

    triangle(vec x, vec y, vec z, vec col){
        a = x;
        b = y;
        c = z;
        color = col;
        transformation = matrix();
    }
    vec geta(){
        return a;
    }
    vec getb(){
        return b;
    }
    vec getc(){
        return c;
    }
    vec clr(){
        return color;
    }
    vec TriCenter() {
        float x_avg = (a.x() + b.x() + c.x()) /3.0;
        float y_avg = (a.y() + b.y() + c.y()) /3.0;
        float z_avg = (a.z() + b.z() + c.z()) /3.0;
        return vec(x_avg,y_avg,z_avg,1);
    }
    matrix getTransformation(){
        return transformation;
    }
    
};

struct Face {
    int vIndex[3];
    int nIndex[3];
//    int tIndex[3];
};


#endif /* triangle_h */
