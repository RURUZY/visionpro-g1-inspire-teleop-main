// Sponge tool handle + backplate, same dimensions as the simulation (mm). Editable version of the STL.
plate = [120, 70, 12];   // backplate x, y, thickness (glue a 120 x 70 x 24 mm sponge underneath)
r = 12;                  // handle radius (24 mm diameter)
top = 124;               // top of the rounded handle above the plate
fillet = 5;              // not in the sim; strengthens the joint
$fn = 96;
translate([-plate[0] / 2, -plate[1] / 2, -plate[2]]) cube(plate);
cylinder(r = r, h = top - r);
translate([0, 0, top - r]) sphere(r = r);
rotate_extrude() difference() {
  square([r + fillet, fillet]);
  translate([r + fillet, fillet]) circle(r = fillet);
}
