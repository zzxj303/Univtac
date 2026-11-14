#define _GNU_SOURCE
#include <dlfcn.h>
#include <fcntl.h>
#include <stdio.h>
#include <unistd.h>
#include <stdarg.h>
#include <errno.h>
#include <sys/stat.h>
#include <stdlib.h>
#include <string.h>
static const char* mapped(const char*p,char*b){const char*prefix="/root/.nvidia-omniverse/logs/";const char*dir=getenv("TH_OMNI_LOG_DIR");if(p&&dir&&!strncmp(p,prefix,strlen(prefix))){snprintf(b,4096,"%s/%s",dir,p+strlen(prefix));return b;}return p;}
static void report(const char *kind,const char *path,int err){if(err==EACCES||err==EPERM){char buf[2048];int n=snprintf(buf,sizeof buf,"IO_DENIED %s %s errno=%d\n",kind,path?path:"(null)",err);write(2,buf,n);}}
int open(const char*p,int f,...){mode_t m=0;if(f&O_CREAT){va_list a;va_start(a,f);m=va_arg(a,int);va_end(a);}int(*fn)(const char*,int,...)=dlsym(RTLD_NEXT,"open");char b[4096];int r=fn(mapped(p,b),f,m),e=errno;if(r<0)report("open",p,e);errno=e;return r;}
int open64(const char*p,int f,...){mode_t m=0;if(f&O_CREAT){va_list a;va_start(a,f);m=va_arg(a,int);va_end(a);}int(*fn)(const char*,int,...)=dlsym(RTLD_NEXT,"open64");char b[4096];int r=fn(mapped(p,b),f,m),e=errno;if(r<0)report("open64",p,e);errno=e;return r;}
int openat(int d,const char*p,int f,...){mode_t m=0;if(f&O_CREAT){va_list a;va_start(a,f);m=va_arg(a,int);va_end(a);}int(*fn)(int,const char*,int,...)=dlsym(RTLD_NEXT,"openat");char b[4096];int r=fn(d,mapped(p,b),f,m),e=errno;if(r<0)report("openat",p,e);errno=e;return r;}
FILE*fopen(const char*p,const char*m){FILE*(*fn)(const char*,const char*)=dlsym(RTLD_NEXT,"fopen");char b[4096];FILE*r=fn(mapped(p,b),m);int e=errno;if(!r)report("fopen",p,e);errno=e;return r;}
FILE*fopen64(const char*p,const char*m){FILE*(*fn)(const char*,const char*)=dlsym(RTLD_NEXT,"fopen64");char b[4096];FILE*r=fn(mapped(p,b),m);int e=errno;if(!r)report("fopen64",p,e);errno=e;return r;}
int mkdir(const char*p,mode_t m){int(*fn)(const char*,mode_t)=dlsym(RTLD_NEXT,"mkdir");char b[4096];int r=fn(mapped(p,b),m),e=errno;if(r<0)report("mkdir",p,e);errno=e;return r;}
