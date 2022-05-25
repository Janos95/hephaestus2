
#![allow(dead_code)]
#![allow(unused_variables)]


//use core::slice::SlicePattern;

use std::cell::RefCell;
use std::rc::Rc;

use std::path::{Path, PathBuf};
use std::ffi::OsStr;
use std::fs::File;
use std::io::{self, prelude::*, BufReader};

use std::ops;


#[derive(Debug, Clone, Copy, PartialEq)]
struct Vec3 {
    x: f64,
    y: f64,
    z: f64,
}

impl Vec3 {
    fn new(x: f64, y: f64, z: f64) -> Vec3 {
        Vec3 { x, y, z }
    }
}

impl ops::Add<Vec3> for Vec3 {
    type Output = Vec3;

    fn add(self, p: Vec3) -> Vec3 {
        Vec3::new(self.x + p.x, self.y + p.y, self.z + p.z)
    }
}

impl ops::Mul<Vec3> for Vec3 {
    type Output = Vec3;

    fn mul(self, p: Vec3) -> Vec3 {
        Vec3::new(self.x * p.x, self.y * p.y, self.z * p.z)
    }
}

impl ops::Div<Vec3> for Vec3 {
    type Output = Vec3;

    fn div(self, p: Vec3) -> Vec3 {
        Vec3::new(self.x / p.x, self.y / p.y, self.z / p.z)
    }
}

impl ops::Sub<Vec3> for Vec3 {
    type Output = Vec3;

    fn sub(self, p: Vec3) -> Vec3 {
        Vec3::new(self.x - p.x, self.y - p.y, self.z - p.z)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Face {
    id : usize,
    mesh : RcMesh,
}

impl Face {
    fn new(id: usize, mesh: RcMesh) -> Face {
        Face { id, mesh }
    }

    fn halfedge(&self) -> Halfedge {
        let id = self.mesh.data.borrow().face_halfedge[self.id] as usize;
        Halfedge { id, mesh: self.mesh.clone() }
    }

    fn is_boundary(&self) -> bool {
        let begin = self.halfedge();
        let mut he = begin.clone();
        loop {
            if he.twin().is_boundary() {
                return true;
            }
            he = he.next();
            if he == begin {
                break;
            }
        }
        false
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Edge {
    id : usize,
    mesh : RcMesh,
}

impl Edge {
    fn new(id: usize, mesh: RcMesh) -> Edge {
        Edge { id, mesh }
    }

    fn halfedge(&self) -> Halfedge {
        Halfedge::new(self.id * 2, self.mesh.clone())
    }

    fn endpoint1(&self) -> Vertex {
        self.halfedge().from()
    }

    fn endpoint2(&self) -> Vertex {
        self.halfedge().to()
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Vertex {
    id : usize,
    mesh : RcMesh,
}

impl Vertex {
    fn new(id: usize, mesh: RcMesh) -> Vertex {
        Vertex { id, mesh }
    }

    fn halfedge(&self) -> Option<Halfedge> {
        let id = self.mesh.data.borrow().vertex_halfedge[self.id];
        if id == std::u32::MAX { None } else { Some(Halfedge::new(id as usize, self.mesh.clone())) }
    }

    fn halfedge_checked(&self) -> Halfedge {
        self.halfedge().expect("Vertex::halfedges() called on isolated vertex")
    }

    fn outgoing_halfedges(&self) -> OutgoingHalfedgeIterator {
        let he = self.halfedge();
        OutgoingHalfedgeIterator { begin: he.clone(), current: he, started: false }
    }

    fn is_boundary(&self) -> bool {
        match self.halfedge() {
            Some(he) => he.is_boundary(),
            None => true,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Halfedge {
    id : usize,
    mesh : RcMesh,
}

impl Halfedge {
    fn new(id: usize, mesh: RcMesh) -> Halfedge {
        Halfedge { id, mesh }
    }

    fn edge(&self) -> Edge {
        Edge::new(self.id / 2, self.mesh.clone())
    }

    fn face(&self) -> Option<Face> {
        let id = self.mesh.data.borrow().halfedge_face[self.id];
        if id == std::u32::MAX {
            None
        } else {
            Some(Face::new(id as usize, self.mesh.clone()))
        }
    }

    fn twin(&self) -> Halfedge {
        Halfedge::new(self.id ^ 1, self.mesh.clone())
    }

    fn next(&self) -> Halfedge {
        let id = self.mesh.data.borrow().next_halfedge[self.id] as usize;
        Halfedge::new(id, self.mesh.clone())
    }

    fn prev(&self) -> Halfedge {
        let id = self.mesh.data.borrow().prev_halfedge[self.id] as usize;
        Halfedge::new(id, self.mesh.clone())
    }

    fn from(&self) -> Vertex {
        let id = self.mesh.data.borrow().halfedge_vertex[self.id] as usize;
        Vertex::new(id, self.mesh.clone())
    }

    fn to(&self) -> Vertex {
        self.twin().from()
    }

    fn is_boundary(&self) -> bool {
        self.face().is_none()
    }
}

struct OutgoingHalfedgeIterator{
    begin : Option<Halfedge>,
    current : Option<Halfedge>,
    started : bool
}

impl Iterator for OutgoingHalfedgeIterator {
  type Item = Halfedge;

  fn next(&mut self) -> Option<Self::Item> {
    if self.started {
        let next = self.current.as_ref().expect("started, so begin is not None").twin().next();
        if next == *self.begin.as_ref().expect("started, so begin is not None") {
            return None;
        }
        self.current = Some(next);
        return self.current.clone();
    } else {
        if let Some(begin) = self.begin.as_ref() {
          self.started = true;
          let current = begin.clone();
          self.current = Some(current.clone());
          return Some(current);
        }
        return None;
    }
  }
}

#[derive(Debug, Clone, PartialEq)]
struct MeshData {
    next_halfedge : Vec<u32>,
    prev_halfedge : Vec<u32>,
    halfedge_face : Vec<u32>,
    halfedge_vertex : Vec<u32>,

    face_halfedge : Vec<u32>,
    vertex_halfedge : Vec<u32>,

    points : Vec<Vec3>,
}

impl MeshData {

    fn new() -> MeshData {
        MeshData { next_halfedge : vec![], 
                prev_halfedge : vec![], 
                halfedge_face : vec![], 
                halfedge_vertex : vec![], 
                face_halfedge : vec![], 
                vertex_halfedge : vec![], 
                points : vec![] 
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Mesh {
   data : RefCell<MeshData>,
}

type RcMesh = Rc<Mesh>;

fn get_extension_from_filename(filename: &str) -> Option<&str> {
    Path::new(filename)
        .extension()
        .and_then(OsStr::to_str)
}

fn find_halfedge(v : &Vertex, w : &Vertex) -> Option<Halfedge> {
    for he in v.outgoing_halfedges() {
        if he.to() == *w {
            return Some(he);
        }
    }
    None
}

fn find_free_gap(he : Halfedge) -> Halfedge {
    let mut he = he;
    loop {
        he = he.next().twin();
        if !he.is_boundary() {
            return he;
        }
    }
}

fn parse_vertex(items : &mut std::str::SplitWhitespace) -> Result<usize, io::Error> {
    if let Some(item) = items.next()  {
        let mut end = item.len();
        if let Some(n) = item.find('/') {
            end = n;
        }
        let id = item[..end].parse::<usize>();
        if let Ok(id) = id {
            return Ok(id);
        }
    }
    Err(io::Error::new(io::ErrorKind::InvalidData, "Cannot parse face: invalid vertex format"))
}

fn try_add_face(mesh : &RcMesh, items : &mut std::str::SplitWhitespace) -> Result<(), io::Error> {
    let i1 = parse_vertex(items)?;
    let i2 = parse_vertex(items)?;
    let i3 = parse_vertex(items)?;
    let face = [Vertex { id : i1 - 1, mesh : mesh.clone() },
                            Vertex { id : i2 - 1, mesh : mesh.clone() },
                            Vertex { id : i3 - 1, mesh : mesh.clone() }];
    println!("Adding face {} {} {}", face[0].id, face[1].id, face[2].id);
    mesh.add_face(&face).unwrap();

    Ok(())
}

impl Mesh {
    fn new() -> Mesh {
        Mesh { data : RefCell::new(MeshData::new()) }
    }

    fn new_shared() -> RcMesh {
        Rc::new(Mesh::new())
    }
    
    pub fn load(path : &str) -> Result<RcMesh, io::Error> {

        if get_extension_from_filename(path) == Some("obj") {
            let file = File::open(path)?;
            let reader = BufReader::new(file);
            let mesh = Mesh::new_shared();
            for line in reader.lines() {
                let line = line?;
                let mut items = line.split_whitespace();
                match items.next() {
                    Some("v") => {
                        let x = items.next().unwrap().parse::<f64>().unwrap();
                        let y = items.next().unwrap().parse::<f64>().unwrap();
                        let z = items.next().unwrap().parse::<f64>().unwrap();
                        mesh.add_vertex(Vec3::new(x, y, z));
                    },
                    Some("f") => {
                       try_add_face(&mesh, &mut items)?;
                    },
                    _ => (),
                }
            }

            println!("Loaded mesh with {} vertices and {} faces", mesh.num_vertices(), mesh.num_faces());
            return Ok(mesh);
        }

        Err(io::Error::new(io::ErrorKind::Other, "Unsupported file type"))
    }

    fn add_face(self : &Rc<Self>, vs : &[Vertex]) -> Option<Face> {
        let n = vs.len();
        assert!(n > 2);

        let mut edge_data : Vec<(Option<Halfedge>, bool, bool)> = Vec::with_capacity(n);

        // test for errors
        for i in 0..n {
            let v = &vs[i];
            if !v.is_boundary() {
                // Nonmanifold vertex
                return None;
            }
            
            let w = &vs[(i + 1) % n]; 
            let he = find_halfedge(&v, &w);
            if he.is_some() && !he.as_ref().unwrap().is_boundary() {
                // Nonmanifold edge
                return None;
            }
            let is_new = he.is_none();
            let needs_adjustment = false;
            edge_data.push((he, is_new, needs_adjustment));
        }

        let mut next_cache = Vec::with_capacity(6*n);

        // re-link patches if necessary
        for i in 0..n {
            let j = (i + 1) % n;
            
            if let (Some(inner_prev), Some(inner_next)) = (&edge_data[i].0, &edge_data[j].0) {
                if inner_prev.next() == *inner_next {
                    let outer_prev = inner_next.twin();
                    let outer_next = inner_prev.twin();

                    let boundary_prev = find_free_gap(outer_prev);
                    let boundary_next = boundary_prev.next();

                    if boundary_prev == *inner_prev {
                        // Patch re-linking failed
                        return None;
                    }

                    assert!(boundary_prev.is_boundary());
                    assert!(boundary_next.is_boundary());

                    let patch_start = inner_prev.next();
                    let patch_end = inner_next.prev();

                    // relink patch
                    next_cache.push((boundary_prev, patch_start));
                    next_cache.push((patch_end, boundary_next));
                    next_cache.push((inner_prev.clone(), inner_next.clone()));
                }
            }
        }

        // Create missing edges 
        for i in 0..n {
            let j = (i + 1) % n;
            let is_new = edge_data[i].1;
            if is_new {
                edge_data[i].0 = Some(self.new_edge(&vs[i], &vs[j]));
            }
        }

        // Create the face
        let f = self.new_face();

        // setup halfedges
        for i in 0..n {
            let j = (i + 1) % n;

            let v = &vs[i];

            let (inner_prev, prev_is_new, _) = edge_data[i].clone();
            let (inner_next, next_is_new, _) = edge_data[j].clone();

            let inner_prev = inner_prev.expect("At this point, all new halfedges should be created");
            let inner_next = inner_next.expect("At this point, all new halfedges should be created");

            // set outer links
            if prev_is_new || next_is_new {

                let outer_prev = inner_next.twin();
                let outer_next = inner_prev.twin();

                match (prev_is_new, next_is_new) {
                    (true, false) => {
                        let boundary_prev = inner_next.prev();
                        next_cache.push((boundary_prev, outer_next.clone()));
                        self.set_vertex_halfedge(v, &outer_next);
                    },
                    (false, true) => {
                        let boundary_next = inner_prev.next();
                        next_cache.push((outer_prev, boundary_next.clone()));
                        self.set_vertex_halfedge(v, &boundary_next);
                    },
                    (true, true) => {
                        if self.data.borrow().vertex_halfedge[v.id] == std::u32::MAX {
                            self.set_vertex_halfedge(v, &outer_next);
                            next_cache.push((outer_prev, outer_next));
                        }
                        else {
                            let boundary_next = v.halfedge().expect("exists by construction");
                            let boundary_prev = boundary_next.prev();
                            next_cache.push((boundary_prev, outer_next));
                            next_cache.push((outer_prev, boundary_next));
                        }
                    },
                    _ => unreachable!(),
                }

                // set inner link
                next_cache.push((inner_prev, inner_next));
            }
            else {
                edge_data[j].2 = v.halfedge().expect("vertex halfedge was set when generating edges") == inner_next;
            }

            // set face handle
            self.set_halfedge_face(edge_data[i].0.as_ref().unwrap(), &f);
        }

        for (he1, he2) in next_cache {
            self.set_next_and_prev(&he1, &he2);
        }

        for ((_, _, needs_adjustment), v) in edge_data.iter().zip(vs) {
            if *needs_adjustment {
                self.adjust_outgoing_halfedge(v);
            }
        }

        Some(f)
    }

    fn add_vertex(self : &Rc<Self>, p : Vec3) -> Vertex {
        let mut data = self.data.borrow_mut();
        let id = data.points.len();
        data.vertex_halfedge.push(std::u32::MAX);
        data.points.push(p);
        Vertex {
            id,
            mesh : self.clone(), 
        }
    }

    fn flip(&mut self, e : &Edge) {

    }

    fn num_vertices(&self) -> usize {
        self.data.borrow().vertex_halfedge.len()
    }

    fn num_faces(&self) -> usize {
        self.data.borrow().face_halfedge.len()
    }

    fn num_edges(&self) -> usize {
        self.data.borrow().halfedge_vertex.len() / 2
    }

    fn num_halfedges(&self) -> usize {
        self.data.borrow().halfedge_vertex.len() 
    }

    // Low level routines

    fn new_edge(self : &Rc<Self>, v1 : &Vertex, v2 : &Vertex) -> Halfedge {
        let mut data = self.data.borrow_mut();
        let n = data.next_halfedge.len();

        let he = Halfedge { id : n + 0, mesh : self.clone() };
        let op = Halfedge { id : n + 1, mesh : self.clone() };

        data.halfedge_face.extend([std::u32::MAX; 2]);
        data.halfedge_vertex.extend([v1.id as u32, v2.id as u32]);
        data.next_halfedge.extend([std::u32::MAX; 2]);
        data.prev_halfedge.extend([std::u32::MAX; 2]);

        he
    }

    fn new_face(self : &Rc<Self>) -> Face {
        let mut data = self.data.borrow_mut();
        let id = data.face_halfedge.len();
        data.face_halfedge.push(std::u32::MAX);
        Face{ id, mesh : self.clone() }
    }

    fn set_vertex_halfedge(self : &Rc<Self>, v : &Vertex, he : &Halfedge) {
        self.data.borrow_mut().vertex_halfedge[v.id] = he.id as u32;
    }

    fn set_next_and_prev(self : &Rc<Self>, he_prev : &Halfedge, he_next : &Halfedge) {
        let mut data = self.data.borrow_mut();
        data.next_halfedge[he_prev.id] = he_next.id as u32;
        data.prev_halfedge[he_next.id] = he_prev.id as u32;
    }

    fn set_halfedge_face(self : &Rc<Self>, he : &Halfedge, f : &Face) {
        self.data.borrow_mut().halfedge_face[he.id] = f.id as u32;
    }

    fn set_halfedge_vertex(self : &Rc<Self>, he : &Halfedge, v : &Vertex) {
        self.data.borrow_mut().halfedge_vertex[he.id] = v.id as u32;
    }

    fn adjust_outgoing_halfedge(self : &Rc<Self>, v : &Vertex) {
        for he in v.outgoing_halfedges() {
            if he.is_boundary() {
                self.set_vertex_halfedge(v, &he);
            }
        }
    }
}

#[test]
fn load_cube() {
    let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.push("assets/cube.obj");
    let mesh = Mesh::load(path.to_str().unwrap()).expect("Failed to load cube.obj");
    assert_eq!(mesh.num_vertices(), 8);
    assert_eq!(mesh.num_faces(), 12);
    assert_eq!(mesh.num_edges(), 18);
}

#[test]
fn load_cube_degenerate() {
    let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.push("assets/cube-degenerate.obj");
    let mesh = Mesh::load(path.to_str().unwrap()).expect("Failed to load cube.obj");
    assert_eq!(mesh.num_vertices(), 8);
    assert_eq!(mesh.num_faces(), 12);
    assert_eq!(mesh.num_edges(), 18);
}

#[test]
fn test_boundary() {
    // Add some vertices
    let mesh = Mesh::new_shared();

    let vs = [
    mesh.add_vertex(Vec3::new(0., 1., 0.)),
    mesh.add_vertex(Vec3::new(1., 0., 0.)),
    mesh.add_vertex(Vec3::new(2., 1., 0.)),
    mesh.add_vertex(Vec3::new(0.,-1., 0.)),
    mesh.add_vertex(Vec3::new(2.,-1., 0.)),
    mesh.add_vertex(Vec3::new(3., 0., 0.)),

    // Single point
    mesh.add_vertex(Vec3::new(0.,-2., 0.))];

    // Add two faces
    let faces = [
    mesh.add_face(&[vs[0].clone(), vs[1].clone(), vs[2].clone()]).unwrap(),
    mesh.add_face(&[vs[1].clone(), vs[3].clone(), vs[4].clone()]).unwrap(),
    mesh.add_face(&[vs[0].clone(), vs[3].clone(), vs[1].clone()]).unwrap(),
    mesh.add_face(&[vs[2].clone(), vs[1].clone(), vs[4].clone()]).unwrap(),
    mesh.add_face(&[vs[5].clone(), vs[2].clone(), vs[4].clone()]).unwrap()];


    /* Test setup:
        0 ==== 2
        |\  0 /|\
        | \  / | \
        |2  1 3|4 5
        | /  \ | /
        |/  1 \|/
        3 ==== 4

        Vertex 6 single
        */


    // Check for boundary vertices
    assert! ( vs[0].is_boundary()  ,"Vertex 0 is not boundary!");
    assert!( !vs[1].is_boundary()  , "Vertex 1 is boundary!");
    assert! ( vs[2].is_boundary()  , "Vertex 2 is not boundary!");
    assert! ( vs[3].is_boundary()  , "Vertex 3 is not boundary!");
    assert! ( vs[4].is_boundary()  , "Vertex 4 is not boundary!");
    assert! ( vs[5].is_boundary()  , "Vertex 5 is not boundary!");
    assert! ( vs[6].is_boundary()  , "Singular Vertex 6 is not boundary!");

    // Check the boundary faces
    assert! ( faces[0].is_boundary() , "Face 0 is not boundary!");
    assert! ( faces[1].is_boundary() , "Face 1 is not boundary!");
    assert! ( faces[2].is_boundary() , "Face 2 is not boundary!");
    assert!( !faces[3].is_boundary() , "Face 3 is boundary!");
    assert! ( faces[4].is_boundary() , "Face 4 is not boundary!");
}