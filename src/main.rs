#![allow(dead_code)]


use core::panic;
use std::str::Chars;

use std::collections::HashMap;
use std::rc::Rc;

use enum_as_inner::EnumAsInner;

#[derive(Debug, Copy, Clone, PartialEq)]
enum Type {
  Bool,
  String,
  Mesh,
}

#[derive(Debug, Clone, PartialEq)]
enum Token {
    LeftParen,
    RightParen,
    End,
    Constant(Value),
    If,
    Else,
    Let,
    Fn,
    EqualSign,
    Arrow,
    Comma,
    Minus,
    Colon,
    Print,
    Return,
    String(String),
    Identifier(String),
    TypeIdentifier(Type),
}

//struct FunctionSignature {
//  arguments : Vec<Type>,
//  ret : Type,
//}
//
//impl FunctionSignature {
//    fn new (args : &[Type], ret : Type) -> Function {
//        FunctionSignature{arguments : args, ret : ret}
//    }
//}
//
//struct Function {
//  name : String,
//  signature : FunctionSignature,
//}
//
//impl Function {
//    fn new (name : &str, args : &[Type], ret : Type) -> Function {
//        Function{name : name, signature : FunctionSignature::new(args, ret)}
//    }
//}

//trait Object {
//  fn name() -> &'static str;
//
//  fn get_methods(&self) -> Vec<Function>;
//
//  fn call_method(&mut self, name : &str, args : &[Value]) -> Option<Value>;
//}

#[derive(Debug, Clone, Copy, PartialEq)]
struct Edge {

}

struct Mesh {

}

impl Mesh {
    fn flip(&mut self, e : Edge) {

    }
}

//struct MeshObject {
//    mesh : Mesh,
//}
//
//impl Object for MeshObject {
//  fn name() -> &'static str {
//    "Mesh"
//  }
//
//  // will need this later for autocomplete
//  fn get_methods(&self) -> Vec<Function> {
//      //let flip_method = Function::new("flip", &[]);
//      vec![]
//  }
//
//  fn call_method(&mut self, name : &str, args : &[Value]) -> Option<Value> {
//      match name {
//        "flip" => {
//            let e = args.first().unwrap().as_edge().unwrap();
//            return Some(self.mesh.flip(e));
//        },
//        _ => panic!("Mesh object does not support the method called")
//      }
//  }
//
//}

#[derive(Debug, Clone, PartialEq, EnumAsInner)]
enum Value {
  Real(f64),
  Int(i64),
  Bool(bool),
  Str(String),
  Edge(Edge),
  //Object(Rc<dyn Object>)
}

struct Tokenizer <'a> {
  it : Chars<'a>,
}

fn contains(a : &mut Chars, keyword: &str) -> bool
{
    for c1 in keyword.chars() {
        if let Some(c2) = a.next() {
           if c1 != c2 {
               return false;
           }
        }
    }
    true
}

impl<'a> Tokenizer<'a> {
    fn new(inp : &'a str) -> Tokenizer<'a> {
        Tokenizer {it : inp.chars()}
    }

    fn parse_type_name(&self, it : &Chars<'a>) -> (Option<Token>, Chars<'a>) {
        for (type_name, type_id) in [
            ("Mesh", Type::Mesh), 
            ("String", Type::String), 
            ("bool", Type::Bool), 
            ] {
            let mut it1 = it.clone();
            if contains(&mut it1, type_name) {
                return (Some(Token::TypeIdentifier(type_id)), it1);
            }
        }
        (None, it.clone())
    }

    fn parse_keyword(&self, it : &Chars<'a>) -> (Option<Token>, Chars<'a>) {
        for (keyword, token) in [
            ("if", Token::If), 
            ("else", Token::Else), 
            ("let", Token::Let), 
            ("fn", Token::Fn), 
            ("return", Token::Return), 
            ("print", Token::Print), 
            ("end", Token::End), 
            ("true", Token::Constant(Value::Bool(true))),
            ("false", Token::Constant(Value::Bool(false)))
            ] {
            let mut it1 = it.clone();
            if contains(&mut it1, keyword) {
                return (Some(token), it1);
            }
        }
        (None, it.clone())
    }

  fn skip_comment(&self, it_old : &Chars<'a>) -> Option<Chars<'a>> {
    let mut it = it_old.clone();
    match (it.next(), it.next()) {
        (Some('/'), Some('/')) => {
            while let Some(c) = it.next() {
                if c == '\n' {
                    break;
                }
            }
            return Some(it);
        },
        (_, _) =>  { return None; },
    }
  }

  fn next_impl(&self) -> (Option<Token>, Chars<'a>) {
    let mut it = self.it.clone();
    let mut prev = self.it.clone();
    while let Some(c) = it.next() {
        if c.is_whitespace() {
            prev = it.clone();
            continue;
        }

        if let Some(it_next) = self.skip_comment(&prev) {
            prev = it_next.clone();
            it = it_next;
            continue;
        }

        if let (Some(token), it) = self.parse_type_name(&prev) {
            return (Some(token), it);
        }

        if let (Some(token), it) = self.parse_keyword(&prev) {
            return (Some(token), it);
        }
         
        match c {
            '(' => return (Some(Token::LeftParen), it),
            ')' => return (Some(Token::RightParen), it),
            '=' => return (Some(Token::EqualSign), it),
            ':' => return (Some(Token::Colon), it),
            '-' => {
                let mut it_next = it.clone();
                match it_next.next() {
                    Some('>') => return (Some(Token::Arrow), it_next),
                    _ => return (Some(Token::Minus), it),
                }
            },
            '"' => {
                let mut s = String::new();
                while let Some(c) = it.next() {
                    if c == '"' {
                        return (Some(Token::String(s)), it);
                    }
                    s.push(c);
                }
                panic!("Unterminated string");
            },
            _ => {
                let mut s = String::new();
                s.push(c);
                while let Some(c) = it.next() {
                    if self.is_identifier_terminating(c) {
                        break;
                    }
                    prev = it.clone();
                    s.push(c);
                }
                return (Some(Token::Identifier(s)), prev);
            }
        }
            
    }
    (None, it)
  }

  fn is_identifier_terminating(&self, c : char) -> bool {
    c.is_whitespace() || c == '(' || c == ')' || c == '=' || c == ':' || c == '-' 
  }

  fn next(&mut self) -> Option<Token> {
    let (token, it) = self.next_impl();
    self.it = it;
    token
  }

  fn peek(&self) -> Option<Token> {
    let (token, _) = self.next_impl();
    token
  }
}

struct Function {
    arg_names : Vec<String>,
    body : Node,
}

struct Environment {
    global_values : HashMap<String, Value>,
    scoped_values : Vec<HashMap<String, Value>>,

    functions : HashMap<String, Function>,
}

impl Environment {
    fn new() -> Environment {
        Environment {
            global_values : HashMap::new(),
            scoped_values : vec![HashMap::new()],
            functions : HashMap::new(),
        }
    }

    fn add_function(&mut self, name : String, func : Function) -> Option<Function> {
        self.functions.insert(name, func)
    }

    fn get_function(&self, name : &str) -> Option<&Function> {
        self.functions.get(name)
    }

    fn get_value(&self, name : &str) -> Option<&Value> {
        if let Some(map) = self.scoped_values.last() {
            if let Some(v) = map.get(name) {
                return Some(v)
            }
        }
        self.global_values.get(name)
    }

    fn set_value(&mut self, name : String, value : Value) -> Option<Value> {
        let map = self.scoped_values.last_mut().unwrap_or(&mut self.global_values);
        map.insert(name, value)
    }

    fn push_scope(&mut self) {
        self.scoped_values.push(HashMap::new());
    }

    fn pop_scope(&mut self) {
        self.scoped_values.pop().expect("No scope to pop");
    }
}

trait AstNode {
    fn evaluate(&self, env : &mut Environment) -> Option<Value>;
}

type Node = Rc<dyn AstNode>;

struct PrintNode {
    expr : Node,
}

struct BlockNode {
    statements : Vec<Node>,
}

struct StringLiteralNode {
    value : String,
}

struct ConstantNode {
    value : Value
}

struct IdentifierNode {
    name : String
}

struct IfNode {
    condition : Node,
    consequent : Node,
    alternate : Option<Node>,
}

struct LetNode {
    pattern : String,
    expression : Node,
}

struct FunctionDefNode {
    name : String,
    body : Node,
}

struct FunctionCallNode {
    func_name : String,
    arguments : Vec<Node>,
}

struct ReturnNode {
    return_value : Option<Node>,
}

impl AstNode for FunctionCallNode {
    fn evaluate(&self, env : &mut Environment) -> Option<Value> {
        env.push_scope();
        let arg_names = env.get_function(&self.func_name).unwrap().arg_names.clone();
        for (arg, arg_name) in self.arguments.iter().zip(arg_names.into_iter()) {
            let arg_value = arg.evaluate(env).expect("Argument expression must evaluate to a value");
            env.set_value(arg_name, arg_value).expect("Argument name must be unique");
        }

        let body = env.get_function(&self.func_name).expect("Calling unkown function").body.clone();
        let result = body.evaluate(env);
        env.pop_scope();
        result
    }
}

impl AstNode for ReturnNode {
    fn evaluate(&self, env : &mut Environment) -> Option<Value> {
        match self.return_value {
            Some(ref node) => node.evaluate(env),
            None => None,
        }
    }
}

impl AstNode for FunctionDefNode {
    fn evaluate(&self, env : &mut Environment) -> Option<Value> {
        None
    }
}

impl AstNode for LetNode {
    fn evaluate(&self, env : &mut Environment) -> Option<Value> {
        let v = self.expression.evaluate(env).expect("let expression requires value");
        let o = env.set_value(self.pattern.clone(), v);
        assert!(o.is_none(), "variable redeclaration");
        None
    }
}

impl AstNode for ConstantNode {
    fn evaluate(&self, env : &mut Environment) -> Option<Value> {
        Some(self.value.clone())
    }
}

impl AstNode for IdentifierNode {
    fn evaluate(&self, env : &mut Environment) -> Option<Value> {
        Some(env.get_value(&self.name).expect("variable not in scope").clone())
    }
}

impl AstNode for IfNode {
    fn evaluate(&self, env : &mut Environment) -> Option<Value> {
        let cond = self.condition.evaluate(env).unwrap();
        match cond {
            Value::Bool(true) => self.consequent.evaluate(env),
            Value::Bool(false) => {
                if let Some(alt) = &self.alternate {
                    alt.evaluate(env)
                } else {
                    None
                }
            }
            _ => panic!("If condition must be a boolean"),
        }
    }
}

impl AstNode for PrintNode {
    fn evaluate(&self, env : &mut Environment) -> Option<Value> {
        let v = self.expr.evaluate(env).unwrap();
        match v {
          Value::Str(s) => println!("{}", s),
          Value::Real(r) => println!("{}", r),
          Value::Int(i) => println!("{}", i),
          _ => println!("TODO: not printable")
        }
        None
    }
}

impl AstNode for BlockNode {
    fn evaluate(&self, env : &mut Environment) -> Option<Value> {
        for s in &self.statements {
            s.evaluate(env);
        }
        None
    }
}

impl AstNode for StringLiteralNode {
    fn evaluate(&self, env : &mut Environment) -> Option<Value> {
        Some(Value::Str(self.value.clone()))
    }
}

struct Parser {
  tokens : Vec<Token>,
  idx : i64,

  env : Environment,
  entry_point : Option<Rc<dyn AstNode>>,
}

fn tokenize(input : &str) -> Vec<Token> {
    let mut tokens = vec![];
    let mut tokenizer = Tokenizer::new(input);
    while let Some(t) = tokenizer.next() {
        tokens.push(t);
    }
    tokens
}

impl Parser {
    fn new(input : &str) -> Parser {
        Parser { tokens : tokenize(input), idx : 0, env : Environment::new(), entry_point: None }
    }

    fn parse_statement(&mut self) -> Option<Node> {
        let t = self.next();
        if let Some(token) = t {
            //println!("statement token {:?}", token);
            match token {
                Token::Print => {
                    self.expect(Token::LeftParen);
                    let expr = self.parse_expression();
                    self.expect(Token::RightParen);
                    return Some(Rc::new(PrintNode { expr }));
                },
                Token::If => {
                    let cond = self.parse_expression();
                    let then_block = self.parse_block();
                    let mut else_block = None;
                    match self.next() {
                        Some(Token::Else) => {
                            else_block = Some(self.parse_block());
                            self.expect(Token::End);
                        },
                        Some(Token::End) => (),
                        _ => panic!("Expected either else or end token, got {:?}", token),
                    }

                    return Some(Rc::new(IfNode {
                        condition : cond,
                        consequent : then_block,
                        alternate : else_block,
                    }));
                },
                Token::Let => {
                    match self.next() {
                        Some(Token::Identifier(name)) => {
                            self.expect(Token::EqualSign);
                            let expr = self.parse_expression();
                            return Some(Rc::new(LetNode{pattern : name, expression : expr}))
                        },
                        _ => panic!("Expected variable name"),
                    }
                },
                Token::Fn => {
                    match self.next() {
                        Some(Token::Identifier(name)) => {
                            self.expect(Token::LeftParen);
                            let mut args = vec![];
                            loop {
                                let (t1,t2,t3) = (self.next(), self.next(), self.next());
                                match (t1, t2, t3) {
                                    (Some(Token::TypeIdentifier(type_id)), Some(Token::Colon), Some(Token::Identifier(arg_name))) => {
                                        args.push(arg_name);
                                    },
                                    (Some(Token::RightParen), _, _) => {
                                        break;
                                    },
                                    (_, _, _) => panic!("Expected type identifier, colon, and identifier"),
                                }
                            }
                            let body = self.parse_block();
                            self.expect(Token::End);
                            let is_entry_point = name == "main";
                            let func_node = Rc::new(FunctionDefNode {
                                name : name.clone(),
                                body : body,
                            });
                            if is_entry_point {
                                self.entry_point = Some(func_node.clone());
                            }
                            self.env.add_function(name, Function{arg_names : args, body : func_node.clone()});
                            return Some(func_node);
                        },
                        _ => panic!("Expected variable name"),
                    }
                }
                Token::Return => {
                    let expr = self.parse_expression();
                    self.expect(Token::End);
                    return Some(Rc::new(ReturnNode { return_value : Some(expr) }));
                },
                // these tokens end a block
                Token::Else => self.go_back(),
                Token::End => self.go_back(),
                _ => panic!("Unexpected token {:?}", token),
            }
        }
        None
    }

    fn go_back(&mut self) {
        assert!(self.idx > 0);
        self.idx = self.idx - 1;
    }

    fn next(&mut self) -> Option<Token> {
        if self.idx >= self.tokens.len() as i64 {
            return None;
        }
        let idx = self.idx as usize;
        self.idx = self.idx + 1;
        Some(self.tokens[idx].clone()) // Instead of cloning could use a ref
    }

    fn peek(&mut self) -> Option<Token> {
        if self.idx >= self.tokens.len() as i64 {
            return None;
        }
        let idx = self.idx as usize;
        Some(self.tokens[idx].clone()) // Instead of cloning could use a ref
    }

    fn parse_expression(&mut self) -> Node {
        let token = self.next();
        //println!("expression token {:?}", token);
        match token {
            Some(Token::LeftParen) => {
                let expr = self.parse_expression();
                self.expect(Token::RightParen);
                return expr;
            },
            Some(Token::String(v)) => {
                return Rc::new(StringLiteralNode { value : v });
            },
            Some(Token::Constant(v)) => {
                return Rc::new(ConstantNode { value : v });
            },
            Some(Token::Identifier(n)) => {
                match self.peek() {
                    Some(Token::LeftParen) => { // function call
                        self.expect(Token::LeftParen);
                        let mut args = vec![];
                        loop {
                            let expr = self.parse_expression();
                            args.push(expr);
                            match self.next() {
                                Some(Token::RightParen) => break,
                                Some(Token::Comma) => (),
                                _ => panic!("Expected either right paren or comma"),
                            }
                        }
                        return Rc::new(FunctionCallNode {
                            func_name : n,
                            arguments : args,
                        });
                    },
                    _ => (),
                }
                return Rc::new(IdentifierNode { name : n });
            },
            _ => panic!("Unexpected token {:?}", token),
        }
    }
    
    fn parse_string_literal(&mut self) -> Node {
        let token = self.next().expect("Expected string literal");
        match token {
            Token::String(s) => Rc::new(StringLiteralNode { value : s }),
            _ => panic!("Unexpected token {:?}", token),
        }
    }

    fn parse_block(&mut self) -> Node {
        let mut statements = Vec::new();
        while let Some(stmt) = self.parse_statement() {
            statements.push(stmt);
        }
        Rc::new(BlockNode{statements})
    }

    fn expect(&mut self, token : Token) {
        let t = self.next();
        if let Some(t) = t {
            if t == token {
                return;
            }
        }
        panic!("Expected token {:?}", token)
    }
}

struct Program {
    root : Rc<dyn AstNode>,
    entry_point : Rc<dyn AstNode>,
    env : Environment,
}

impl Program {
    fn new(script : &str) -> Program {
        let mut parser = Parser::new(script);
        let root = parser.parse_block();
        Program{root : root, entry_point : parser.entry_point.unwrap(), env : parser.env}
    }

    fn run (&mut self) {
        self.entry_point.evaluate(&mut self.env);
    }
}

fn main() {
  let script = include_str!("import_mesh.hep");

  //let mut tokenizer = Tokenizer::new(script);
  //while let Some(token) = tokenizer.next() {
  //    println!("{:?}", token);
  //}

  let mut program = Program::new(script);
  program.run();
}

