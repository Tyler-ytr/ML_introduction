#include <iostream>
#include <algorithm>

#include<cmath>
using namespace std;

int a[11]={24,53,25,32,52,22,43,48} ;
int b[11]={40,52,77,48,110,38,44,65};
int c[11]={1,0,1,1,1,1,0,1};

int main()
{
	int chosen_one;
	int D1[11],D2[11];
	int cnt_d1=0,cnt_d2=0;
	double dini_index[11];
	double max1=-100000000;
		
	
	
	for(int i=0;i<8;i++)
	{
		cnt_d1=0;
		cnt_d2=0;
		for(int j=0;j<8;j++)
		{
			if(b[j]>=b[i])
			{
				D1[cnt_d1]=j;
				cnt_d1++;
			}
			else
			{
				D2[cnt_d2]=j;
				cnt_d2++;
			}
		}
		cout<<"_______"<<i<<endl;
		for(int t=0;t<cnt_d1;t++)
		{
			cout<<D1[t]<<endl;;
		}
		cout<<"sdsdsd"<<endl;
			for(int t=0;t<cnt_d2;t++)
		{
			cout<<D2[t]<<endl;;
		}
		cout<<"_______"<<endl;
		
		
		double sum1=0,sum2=0;
		for(int t=0;t<cnt_d1;t++)
		{
			sum1+=c[D1[t]];
		}
		for(int t=0;t<cnt_d2;t++)
		{
			sum2+=c[D2[t]];
		}
		
		cout<<"cnt "<<i<<":"<<cnt_d1<<" "<<cnt_d2<<endl;
		double p11=0;
		if(cnt_d1!=0)
		{
			p11=sum1/cnt_d1;
		}
		else p11=0;
		
		double p10=1-p11;
		
		double p21;
		if(cnt_d2!=0)
		{
			p21=sum2/cnt_d2;
		}
		else p21=0;
		
		double p20=1-p21;
		
//		double dini1=1-p11*p11-p10*p10;
//		double dini2=1-p21*p21-p20*p20;
double dini1;
double dini2;

if(p11==0)
{
	dini1=-log(p10)/log(2)*p10;
}
if(p10==0)
{
	dini1=-log(p11)/log(2)*p11;
}
if(p11==0||p10==0)
{
	;
}
else
dini1=-log2(p11)*p11-log2(p10)*p10;
cout<<i<<"dini1"<<dini1<<endl;
if(p21==0)
{
	dini2=-log(p20)/log(2)*p20;
}
if(p20==0)
{
	dini2=-log(p21)/log(2)*p21;
}
if(p21==0||p20==0)
{
	;
}
else
dini2=-(log(p21)/log(2))*p21-log(p20)/log(2)*p20;
//double dini1=-log(p11)/log(2)*p11-log(p10)/log(2)*p10;
//double dini2=-log(p21)/log(2)*p21-log(p21)/log(2)*p21;
//if(p11==0||p10==0)dini1=1000000000;
//if(p21==0||p20==0)dini2=1000000000;
//if(i==6)cout<<i<<" "<<p11<<" "<<p10<<" "<<p21<<" "<<p20<<" "<<endl;
		cout<<i<<" "<<p11<<" "<<p10<<" "<<p21<<" "<<p20<<" "<<" cnt:"<<cnt_d1<<" "<<cnt_d2<<endl;
		dini_index[i]=0.321982-((double)cnt_d1/8)*dini1-(double)cnt_d2/8*dini2;
		cout<<i<<":"<<dini1<<" "<<dini2<<" "<<dini_index[i]<<" "<<-((double)cnt_d1/10)*dini1-(double)cnt_d2/10*dini2<<endl; 
		max1=max(max1,dini_index[i]);
		
	}
	cout<<endl;
	for(int i=0;i<8;i++)
	{
		if(max1==dini_index[i])
		cout<<"Max Gain :"<<b[i]<<" "<<dini_index[i]<<endl;
	}
cout<<endl;
		for(int i=0;i<8;i++)
	{
		
		cout<<i<<":"<<"x1[i]: "<<b[i]<<" "<<dini_index[i]<<endl;
	}
//	int chosen_one;
//	int D1[11],D2[11];
//	int cnt_d1=0,cnt_d2=0;
//	double dini_index[11];
//	double min1=1000000;
//		
//	
//	
//	for(int i=0;i<10;i++)
//	{
//		cnt_d1=0;
//		cnt_d2=0;
//		for(int j=0;j<10;j++)
//		{
//			if(b[j]<=b[i])
//			{
//				D1[cnt_d1]=j;
//				cnt_d1++;
//			}
//			else
//			{
//				D2[cnt_d2]=j;
//				cnt_d2++;
//			}
//		}
//		
//		double sum1=0,sum2=0;
//		for(int t=0;t<cnt_d1;t++)
//		{
//			sum1+=c[D1[t]];
//		}
//		for(int t=0;t<cnt_d2;t++)
//		{
//			sum2+=c[D2[t]];
//		}
//		
//		cout<<"cnt "<<i<<":"<<cnt_d1<<" "<<cnt_d2<<endl;
//		double p11=0;
//		if(cnt_d1!=0)
//		{
//			p11=sum1/cnt_d1;
//		}
//		else p11=0;
//		
//		double p10=1-p11;
//		
//		double p21;
//		if(cnt_d2!=0)
//		{
//			p21=sum2/cnt_d2;
//		}
//		else p21=0;
//		
//		double p20=1-p21;
//		
//		double dini1=1-p11*p11-p10*p10;
//		double dini2=1-p21*p21-p20*p20;
//		
//		dini_index[i]=((double)cnt_d1/10)*dini1+(double)cnt_d2/10*dini2;
//		cout<<i<<":"<<dini1<<" "<<dini2<<" "<<dini_index[i]<<" "<<(cnt_d1/10)*dini1+cnt_d2/10*dini2<<endl; 
//		min1=min(min1,dini_index[i]);
//		
//	}
//	cout<<endl;
//	for(int i=0;i<10;i++)
//	{
//		if(min1==dini_index[i])
//		cout<<i<<":"<<b[i]<<endl;
//	}
//	
//	
//	
//	cout<<min1<<endl;
	

	
	
	
	
	
	return 0;
}
