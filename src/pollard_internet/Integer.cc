/***************************** BEGIN ********************************/

#include <stdlib.h>
#include "Integer.h"

//
// Constructors
//
Integer::Integer()		{mpz_init_set_ui(_x, 0); base=10;}
Integer::Integer(int a)		{mpz_init_set_si(_x,a);base=10;}
Integer::Integer(unsigned int a){mpz_init_set_ui(_x, a); base=10;}
Integer::Integer(const Integer &a) {mpz_init_set(_x,a._x);base=a.base;}
Integer::Integer(char * a)	{mpz_init_set_str(_x,a,10);base=10;}
Integer::Integer(mpz_t a)	{mpz_init_set(_x, a);base=10;}
//
// Destructor
//
Integer::~Integer()	{mpz_clear(_x); }	// releases memory
//
// Simple access functions
//
int	Integer::getbase(void)	{ return base; }
void	Integer::setbase(int b)	{ base = b; }
//
// Casting
//
Integer::operator bool(void)	{ return (bool) (mpz_cmp_ui(_x,0))? 1 :0; }
Integer::operator int(void)	{ return (int)  mpz_get_si(_x); }
Integer::operator unsigned int(void)	{ return (unsigned int) mpz_get_ui(_x); }
Integer::operator float(void)	{ return (float) mpz_get_d(_x); }
Integer::operator double(void)	{ return (double) mpz_get_d(_x); }
//
// Assignments
//
void Integer::operator =(Integer a)	{ mpz_set(_x, a._x); base=a.base;}
void Integer::operator =(int a)		{ mpz_set_si(_x, a); base=10;}
void Integer::operator =(unsigned int a){ mpz_set_ui(_x, a); base=10;}
void Integer::operator =(char *a)	{ mpz_set_str(_x, a, 10); base=10;}
void Integer::operator =(const char *a)	{ mpz_set_str(_x, a, 10); base=10;}
void Integer::operator =(mpz_t a)	{ mpz_set(_x, a); base=10; }
//
// Unary Operators -, !, etc...
//
Integer Integer::operator -(void) { Integer r; mpz_neg(r._x, _x); return r; } 
bool	Integer::operator !(void) { return (mpz_cmp_ui(_x, 0)==0)? 1 : 0; }
void	Integer::operator +=(Integer a)		{ mpz_add(_x, _x, a._x); }
void	Integer::operator +=(int a)		{ (a>0)? mpz_add_ui(_x, _x, a) : mpz_sub_ui(_x, _x, -a); }
void	Integer::operator +=(unsigned int a)	{ mpz_add_ui(_x, _x, a); }
void	Integer::operator -=(Integer a)		{ mpz_sub(_x, _x, a._x); }
void	Integer::operator -=(int a)		{ (a>0)? mpz_sub_ui(_x, _x, a) : mpz_add_ui(_x, _x, -a); }
void	Integer::operator -=(unsigned int a)	{ mpz_sub_ui(_x, _x, a); }
void	Integer::operator *=(Integer a)		{ mpz_mul(_x, _x, a._x); }
void	Integer::operator *=(int a)		{ mpz_mul_ui(_x, _x,(a>0)? a :-a); if (a<0) mpz_neg(_x,_x);}
void	Integer::operator *=(unsigned int a)	{ mpz_mul_ui(_x, _x, a); }
void	Integer::operator /=(Integer a)		{ mpz_tdiv_q(_x, _x, a._x); }
void	Integer::operator /=(int a)		{ mpz_tdiv_q_ui(_x, _x,(a>0)? a:-a); if (a<0) mpz_neg(_x,_x);}
void	Integer::operator /=(unsigned int a)	{ mpz_tdiv_q_ui(_x, _x, a); }
void	Integer::operator %=(Integer a)		{ mpz_mod(_x, _x, a._x); }
void	Integer::operator %=(int a)		{ mpz_mod_ui(_x, _x, a); }
void	Integer::operator ^=(int a)		{ mpz_pow_ui(_x, _x, a); }
//
// Inc/Dec --  prefix:  ++a
Integer	Integer::operator ++(void) { mpz_add_ui(_x, _x, 1); return _x;}
Integer	Integer::operator --(void) { mpz_sub_ui(_x, _x, 1); return _x;}
//
// Inc/Dec -- postfix: a++
Integer	Integer::operator ++(int a) {Integer r=_x;mpz_add_ui(_x,_x,1); return r; }
Integer	Integer::operator --(int a) {Integer r=_x;mpz_sub_ui(_x,_x,1); return r;}
//
//  Binary (friendly) operators
//
Integer	operator +(Integer a, Integer b) {Integer c;mpz_add(c._x,a._x,b._x);return c;}
Integer	operator +(Integer a, int b) {Integer c; if (b>0) mpz_add_ui(c._x,a._x,b); else mpz_sub_ui(c._x, a._x, -b); return c;}
Integer	operator +(int a, Integer b) { return b+a; }
Integer	operator -(Integer a, Integer b) {Integer c;mpz_sub(c._x,a._x,b._x);return c;}
Integer	operator -(Integer a, int b) { return a + (-b);}
Integer	operator -(int a, Integer b) { return -(b-a); }
Integer	operator *(Integer a, Integer b) {Integer c; mpz_mul(c._x, a._x, b._x); return c; }
Integer	operator *(Integer a, int b) {Integer c; mpz_mul_ui(c._x, a._x, (b>0)? b:-b); if (b<0) return -c; else return c; }
Integer	operator *(int a, Integer b) { return b*a; }
Integer	operator /(Integer a, Integer b) {Integer c; mpz_tdiv_q(c._x, a._x, b._x); return c; }
Integer	operator /(Integer a, int b) {Integer c; mpz_tdiv_q_ui(c._x, a._x, (b>0)? b:-b); if (b<0) return -c; else return c; }
Integer	operator /(int a, Integer b) {Integer c; c = a; return c/b; }
Integer	operator %(Integer a, Integer b) {Integer c; mpz_mod(c._x, a._x, b._x); return c; }
Integer	operator %(Integer a, int b) {Integer c; mpz_mod_ui(c._x, a._x, b); return c; }
Integer	operator %(int a, Integer b) {Integer c; c = a; return c%b; }
Integer	operator ^(Integer a, int b) {Integer c; mpz_pow_ui(c._x, a._x, b); return c; }
Integer	operator &(Integer a, Integer b) {Integer c; mpz_and(c._x, a._x, b._x); return c; }
Integer	operator |(Integer a, Integer b) {Integer c; mpz_ior(c._x, a._x, b._x); return c; }
bool operator ==(Integer a, Integer b) {return (mpz_cmp(a._x, b._x)==0)? 1:0;}
bool operator ==(Integer a, int b) {return (mpz_cmp_si(a._x, b)==0)? 1:0;}
bool operator ==(int a, Integer b) {return (b==a); }
bool operator !=(Integer a, Integer b) {return (mpz_cmp(a._x, b._x)!=0)? 1:0;}
bool operator !=(Integer a, int b) {return (mpz_cmp_si(a._x, b)!=0)? 1:0;}
bool operator !=(int a, Integer b) {return (b!=a); }
bool operator <(Integer a, Integer b) {return (mpz_cmp(a._x, b._x)<0)? 1:0;}
bool operator <(Integer a, int b) {return (mpz_cmp_si(a._x, b)<0)? 1 : 0;}
bool operator <(int a, Integer b) {return (b>a); }
bool operator <=(Integer a,Integer b) {return (mpz_cmp(a._x, b._x)>0)? 0:1;}
bool operator <=(Integer a, int b) {return (mpz_cmp_si(a._x, b)>0)? 0 : 1;}
bool operator <=(int a, Integer b) {return (b>=a); }
bool operator >(Integer a, Integer b) {return (mpz_cmp(a._x, b._x)>0)? 1:0;}
bool operator >(Integer a, int b) {return (mpz_cmp_si(a._x, b)>0)? 1 : 0;}
bool operator >(int a, Integer b) {return (b<a); }
bool operator >=(Integer a, Integer b) {return (mpz_cmp(a._x, b._x)<0)? 0:1;}
bool operator >=(Integer a, int b) {return (mpz_cmp_si(a._x, b)<0)? 0 : 1;}
bool operator >=(int a, Integer b) {return (b<=a); }
ostream& operator <<(ostream& s, Integer i) {
	char *p; p = (char *) new char[4+logn(i, i.getbase() )];
	Int2a(i, p); s << p; delete p; return s; }
istream& operator >>(istream& s, Integer &a) {
	char *p; p = (char *) new char[8192];
	s >> p; a = (Integer) p; delete p; return s; }
Integer& operator <<(Integer a, int b) {
	Integer c; c=a; mpz_mul_2exp(a._x, c._x, b); return a;
	}
Integer& operator >>(Integer a, int b) {
	Integer c; c=a; mpz_tdiv_q_2exp(a._x, c._x, b); return a;
	}
//
// Misc Number Theory Fuctions  (friendly)
//
void	Int2a(Integer i, char *a) { mpz_get_str(a, i.base, i._x ); }
Integer	gcd(Integer a, Integer b) {
	Integer g; mpz_gcd(g._x, a._x, b._x); return g;}
Integer	exgcd(Integer a, Integer b, Integer &c, Integer &d) {
	Integer g; mpz_gcdext(g._x, c._x, d._x, a._x, b._x); return g; }
Integer InvModN(Integer a, Integer n) {
	Integer c; mpz_invert(c._x, a._x, n._x); return c%n; }
Integer PowModN(Integer a, int b, Integer n) {
	Integer c; mpz_powm_ui(c._x, a._x, b, n._x); return c%n; }
Integer PowModN(Integer a, Integer b, Integer n) {
	Integer c; mpz_powm(c._x, a._x, b._x, n._x); return c%n; }
bool	isprime(Integer a) 	{ return (bool) mpz_probab_prime_p(a._x, 25); }
bool	issquare(Integer a)	{ return (bool) mpz_perfect_square_p(a._x); }
bool	testbit(Integer a, unsigned long int i) {
	return (i == mpz_scan1(a._x, i))? 1 : 0; }
Integer sqrt(Integer a)	{ Integer r; mpz_sqrt(r._x, a._x); return r; }
int	digits(Integer a, int b) { return mpz_sizeinbase(a._x, b); }

/***** I don't think these two functions are too reliable... -Paul *****/
int	Jacobi(Integer a, Integer b) { return mpz_jacobi(a._x, b._x);}
int	Legendre(Integer a, Integer b) { return mpz_legendre(a._x, b._x);}
/***********************************************************************/
Integer random(Integer a) {
	Integer r; mp_size_t siz; long rr; time_t t;

	srand48((long)time(&t));	// set seed
	rr = (lrand48()&127) + 1;	// get a random number
	siz = (log2(a)+1)*3;
	if (siz<1) siz = 1;		// estimate number of limbs
	while(rr--) mpz_random(r._x, siz);	// get random generator goin
	r %= a;
	return r;
}
//
//	Non friendly (miscellaneous) useful functions
//
Integer	lcm(Integer a, Integer b) { return (a*b)/gcd(a,b); }
Integer	LCM(Integer n) {	// this is lcm[1, 2, 3, ... , n]
	if (n < 3) return 2;
	else	   return lcm(LCM(n-1), n); }
int	logn(Integer a, int b) 	{ return digits(a, b) - 1; }
int	log2(Integer a)		{ return logn(a, 2); }
Integer fact(Integer a)		{
	Integer r=1, i=1; while (i++ < a) r *= i; return r;}
Integer P_n_k(Integer n, Integer k)	{
	Integer r=1, i=k; while (i++ < n) r *= i; return r;}
Integer C_n_k(Integer n, Integer k)	{
	Integer r=1, i=1; while (i++ <= k) {r *= (n-(i-2)); r /= i-1;}
	return r;}

Integer nextprime(Integer a)		{
	if (a%2 == 0) { a++; if (isprime(a)) return a; }
	do {a++; a++;} while (!isprime(a)); return a;}
Integer prevprime(Integer a)		{
	if (a<3) return a;
	if (a%2 == 0) { a--; if (isprime(a)) return a; }
	do {a--; a--;} while (!isprime(a) && a>1); return a;}

Integer euler(Integer n) {	// there are better ways to do this
	Integer count, i;
	count = 1;
	for (i=2; i<n; i++) if (gcd(i, n) == 1) count++;
	return count;
}

// Quick sneaky prototype
Integer SqrtShanks(Integer x, Integer m);

Integer SqrtModN(Integer x, Integer p) {
	return SqrtShanks(x, p);
}

Integer SqrtShanks(Integer x, Integer m) {
	// square root mod a small prime by Shanks method
	// returns 0 if root does not exist or m not prime
    Integer z,y,v,w,t,q,r;
    int i,e,n;
    bool pp;
    x %= m;
    if (x==0) return 0;
    if (x==1) return 1;
//    if (PowModN(x,(m-1)/2,m)!=1) return 0;	// Legendre symbol not 1
    if (Legendre(x,m)!=1) return 0;		// Legendre symbol not 1
    if (m%4==3) return PowModN(x,(m+1)/4,m);	// easy case for m=4.k+3
    if (m%8==5) {				// also relatively easy
	t = PowModN(x,(m-1)/4,m);
	if (t==1) return PowModN(x,(m+3)/8,m);
	if (t==(m-1)) {
	    t = (4*x)%m;			// muldiv((small)4,x,(small)0,m,&t);
	    t = PowModN(t,(m+3)/8,m);
	    t = (t*(m+1)/2)%m;			// muldiv(t,(m+1)/2,(small)0,m,&t);
	    return t;
	    }
	return 0;
	}
    q=m-1;
    e=0;
    while (q%2==0) {
	q/=2;
	e++;
	}
    if (e==0) return 0;      	// even m 
    for (r=2;;r++) {		// find suitable z
	z=PowModN(r, q, m);
	if (z==1) continue;
	t=z;
	pp=0;
	for (i=1;i<e;i++) {		// check for composite m
	    if (t==(m-1)) pp=1;
	    t = (t*t)%m;		// muldiv(t,t,(small)0,m,&t);
	    if (t==1 && !pp) return 0;
	    }
	if (t==(m-1)) break;
	if (!pp) return 0;   /* m is not prime */
    }
    y=z;
    r=e;
    v=PowModN(x,(q+1)/2,m);
    w=PowModN(x,q,m);
    while (w!=1) {
	t=w;
	for (n=0;t!=1;n++) t = (t*t)%m;		// muldiv(t,t,(small)0,m,&t);
	if (n>=r) return 0;
	y=PowModN(y, (1<<((int)r-n-1)), m);
	v = (v*y)%m;				// muldiv(v,y,(small)0,m,&v);
	y = (y*y)%m;				// muldiv(y,y,(small)0,m,&y);
	w = (w*y)%m;				// muldiv(w,y,(small)0,m,&w);
	r=n;
    }
    return (Integer)v;
}
