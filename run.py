import os
import subprocess
import argparse

def docker_run(cmd): 
    uid = int(subprocess.run(['id', '-u'], stdout=subprocess.PIPE).stdout)
    gid = int(subprocess.run(['id', '-g'], stdout=subprocess.PIPE).stdout)
    subprocess.run(['docker', 'build',  '-t', 'convex_adversarial', '.'])


    s = """docker run -it --runtime=nvidia --rm -w /home -v ${{PWD}}/nips/:/home/nips/ convex_adversarial zsh -c '{}; {}'"""
    fix_permissions = 'chown -R {}:{} /home/nips/'.format(uid, gid)

    os.system(s.format(cmd, fix_permissions))

if __name__ == "__main__":
    uid = int(subprocess.run(['id', '-u'], stdout=subprocess.PIPE).stdout)
    gid = int(subprocess.run(['id', '-g'], stdout=subprocess.PIPE).stdout)

    parser = argparse.ArgumentParser()
    parser.add_argument('cmd', type=str, help='command to run')
    
    args = parser.parse_args()
    docker_run(args.cmd)