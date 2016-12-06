c
c
      subroutine eebls(n,t,x,u,v,nf,fmin,df,nb,qmi,qma,
     &p,bper,bpow,depth,qtran,in1,in2)
c
c------------------------------------------------------------------------
c     >>>>>>>>>>>> This routine computes BLS spectrum <<<<<<<<<<<<<<
c
c         [ see Kovacs, Zucker & Mazeh 2002, A&A, Vol. 391, 369 ]
c
c     This is the slightly modified version of the original BLS routine
c     by considering Edge Effect (EE) as suggested by
c     Peter R. McCullough [ pmcc@stsci.edu ].
c
c     This modification was motivated by considering the cases when
c     the low state (the transit event) happened to be devided between
c     the first and last bins. In these rare cases the original BLS
c     yields lower detection efficiency because of the lower number of
c     data points in the bin(s) covering the low state.
c
c     For further comments/tests see  www.konkoly.hu/staff/kovacs.html
c------------------------------------------------------------------------
c
c     Input parameters:
c     ~~~~~~~~~~~~~~~~~
c
c     n    = number of data points
c     t    = array {t(i)}, containing the time values of the time series
c     x    = array {x(i)}, containing the data values of the time series
c     u    = temporal/work/dummy array, must be dimensioned in the
c            calling program in the same way as  {t(i)}
c     v    = the same as  {u(i)}
c     nf   = number of frequency points in which the spectrum is computed
c     fmin = minimum frequency (MUST be > 0)
c     df   = frequency step
c     nb   = number of bins in the folded time series at any test period
c     qmi  = minimum fractional transit length to be tested
c     qma  = maximum fractional transit length to be tested
c
c     Output parameters:
c     ~~~~~~~~~~~~~~~~~~
c
c     p    = array {p(i)}, containing the values of the BLS spectrum
c            at the i-th frequency value -- the frequency values are
c            computed as  f = fmin + (i-1)*df
c     bper = period at the highest peak in the frequency spectrum
c     bpow = value of {p(i)} at the highest peak
c     depth= depth of the transit at   *bper*
c     qtran= fractional transit length  [ T_transit/bper ]
c     in1  = bin index at the start of the transit [ 0 < in1 < nb+1 ]
c     in2  = bin index at the end   of the transit [ 0 < in2 < nb+1 ]
c
c
c     Remarks:
c     ~~~~~~~~
c
c     -- *fmin* MUST be greater than  *1/total time span*
c     -- *nb*   MUST be lower than  *nbmax*
c     -- Dimensions of arrays {y(i)} and {ibi(i)} MUST be greater than
c        or equal to  *nbmax*.
c     -- The lowest number of points allowed in a single bin is equal
c        to   MAX(minbin,qmi*N),  where   *qmi*  is the minimum transit
c        length/trial period,   *N*  is the total number of data points,
c        *minbin*  is the preset minimum number of the data points per
c        bin.
c
c========================================================================
c
      implicit none
c

      integer, intent(in) :: n, nf, nb
      real*8, dimension(n), intent(in) :: t, x
      real*8, dimension(n), intent(inout) :: u, v
      real*8, intent(in) :: fmin, df, qmi, qma

      real*8, dimension(nf), intent(out) :: p
      real*8, intent(out) :: bper, bpow, depth, qtran
      integer, intent(out) :: in1, in2

      real*8, dimension(2000) :: y
      integer, dimension(2000) :: ibi
      integer :: minbin,nbmax,kmi,kma,kkmi,nb1,nbkma,i,jf,j,jnb,k,kk,
     &nb2,jn1,jn2
      real*8 :: rn,tot,s,t1,f0,p0,ph,power,rn1,pow,rn3,s3

c
      minbin = 5
      nbmax  = 2000
      if(nb.gt.nbmax) write(*,*) ' NB > NBMAX !!'
      if(nb.gt.nbmax) stop
      tot=t(n)-t(1)
      if(fmin.lt.1.0d0/tot) write(*,*) ' fmin < 1/T !!'
      if(fmin.lt.1.0d0/tot) stop
c------------------------------------------------------------------------
c
      rn=dfloat(n)
      kmi=idint(qmi*dfloat(nb))
      if(kmi.lt.1) kmi=1
      kma=idint(qma*dfloat(nb))+1
      kkmi=idint(rn*qmi)
      if(kkmi.lt.minbin) kkmi=minbin
      bpow=0.0d0
c
c     The following variables are defined for the extension
c     of arrays  ibi()  and  y()  [ see below ]
c
      nb1   = nb+1
      nbkma = nb+kma
c
c=================================
c     Set temporal time series
c=================================
c
      s=0.0d0
      t1=t(1)
      do 103 i=1,n
      u(i)=t(i)-t1
      s=s+x(i)
 103  continue
      s=s/rn
      do 109 i=1,n
      v(i)=x(i)-s
 109  continue
c
c******************************
c     Start period search     *
c******************************
c
      do 100 jf=1,nf
      f0=fmin+df*dfloat(jf-1)
      p0=1.0d0/f0
c
c======================================================
c     Compute folded time series with  *p0*  period
c======================================================
c
      do 101 j=1,nb
        y(j) = 0.0d0
      ibi(j) = 0
 101  continue
c
      do 102 i=1,n
      ph     = u(i)*f0
      ph     = ph-idint(ph)
      j      = 1 + idint(nb*ph)
      ibi(j) = ibi(j) + 1
        y(j) =   y(j) + v(i)
 102  continue
c
c-----------------------------------------------
c     Extend the arrays  ibi()  and  y() beyond
c     nb   by  wrapping
c
      do 104 j=nb1,nbkma
      jnb    = j-nb
      ibi(j) = ibi(jnb)
        y(j) =   y(jnb)
 104  continue
c-----------------------------------------------
c
c===============================================
c     Compute BLS statistics for this period
c===============================================
c
      power=0.0d0
c
      do 1 i=1,nb
      s     = 0.0d0
      k     = 0
      kk    = 0
      nb2   = i+kma
      do 2 j=i,nb2
      k     = k+1
      kk    = kk+ibi(j)
      s     = s+y(j)
      if(k.lt.kmi) go to 2
      if(kk.lt.kkmi) go to 2
      rn1   = dfloat(kk)
      pow   = s*s/(rn1*(rn-rn1))
      if(pow.lt.power) go to 2
      power = pow
      jn1   = i
      jn2   = j
      rn3   = rn1
      s3    = s
 2    continue
 1    continue
c
      power = dsqrt(power)
      p(jf) = power
c
      if(power.lt.bpow) go to 100
      bpow  =  power
      in1   =  jn1
      in2   =  jn2
      qtran =  rn3/rn
      depth = -s3*rn/(rn3*(rn-rn3))
      bper  =  p0
c
 100  continue
c
c     Edge correction of transit end index
c
      if(in2.gt.nb) in2 = in2-nb
c
      return
      end
c
c
