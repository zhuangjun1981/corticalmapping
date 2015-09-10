from imaging_behavior.io import decimate_JCamF as deci
import os
import datetime
import shutil

def send_email(to,message):
    """ Mcafee access scan must be disabled for this to work!"""
    import smtplib

    gmail_user = "neuralcodingbehavior@gmail.com"
    gmail_pwd = "virtualforager"
    FROM = 'neuralcodingbehavior@gmail.com'
    TO = [to]#must be a list
    SUBJECT = "Auto-generated message"
    TEXT = message

    # Prepare actual message
    message = """\From: %s\nTo: %s\nSubject: %s\n\n%s
    """ % (FROM, ", ".join(TO), SUBJECT, TEXT)
    try:
        #server = smtplib.SMTP(SERVER) 
        server = smtplib.SMTP("smtp.gmail.com", 587) #or port 465 doesn't seem to work!
        server.ehlo()
        server.starttls()
        server.login(gmail_user, gmail_pwd)
        server.sendmail(FROM, TO, message)
        #server.quit()
        server.close()
        print 'successfully sent the mail'
    except Exception, e:
        print "failed to send mail"
        print e



source = r'D:/'
JCamfiles = []
#Get a list of all large jcam files
# the size limitation will exclude single frame vascular images
for f in os.listdir(source):
    if "JCam" in f and os.path.getsize(os.path.join(source,f)) > 20000000:
        JCamfiles.append(f)

#Decimate, then delete the original
for f in JCamfiles:
	# print os.path.join(source,f)
	deci(os.path.join(source,f),verbose=True,spatial_compression=4,temporal_compression=1)
	print ""
	print "DELETING ",os.path.join(source,f)
	os.remove(os.path.join(source,f))

#Create a destination folder with a timestamp. It's up to me to manually sort through these later and make sure that all of the subordinate files are moved to the right place
destination = r'\\aibsdata2\nc-ophys\imagedata\doug\unsorted'
datestamp = datetime.datetime.now().strftime("%Y"+"-"+"%m"+"-"+"%d"+"--"+"%H"+"%M")
os.makedirs(os.path.join(destination,datestamp))

# move the files
for f in os.listdir(source):
	if "$" not in f and "System" not in f:
		print "Moving: ",os.path.join(source,f), "to", os.path.join(destination,datestamp,f)
		shutil.move(os.path.join(source,f),os.path.join(destination,datestamp,f))

# send an email when done
print "sending email"
msg = "Drive is clear"
send_email(to='dougo@alleninstitute.org',message=msg)

